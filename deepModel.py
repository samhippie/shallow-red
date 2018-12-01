#!/usr/bin/env python3

import io
import numpy as np
import psycopg2
import sys
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import modelInput


#this could go in a config file or something
dbConnect = "dbname='shallow-red' user='shallow-red' host='localhost' password='shallow-red'"


#model of the network
class Net(nn.Module):
    def __init__(self, softmax=False, width=1000):
        super(Net, self).__init__()

        self.softmax = softmax

        #simple feed forward
        #this is kind of big, but I waste CPU cycles
        #with smaller networks (given mini-batching)
        #and this should be even more true with a GPU
        self.fc1 = nn.Linear(modelInput.stateSize, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc4 = nn.Linear(width, width)
        self.fc5 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, modelInput.numActions)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        #normalize to 0 mean and unit variance
        #like in the paper
        #x = self.normalizer(x)
        if self.softmax:
            x = F.softmax(self.fc6(x), dim=0)
        else:
            x = self.fc6(x)
        return x

class DeepCfrModel:

    #for advantages, the input is the state vector
    #and the output is a vector of each move's advantage
    #for strategies, the input is the state vector
    #and the output is a vector of each move's probability

    #so the inputs are exactly the same (modelInput.stateSize), and the outputs
    #are almost the same (modelInput.numActions)
    #strategy is softmaxed, advantage is not



    def __init__(self, name, softmax, lr=0.001, sampleCacheSize=10000, clearDb=True):
        self.softmax = softmax
        self.lr = lr

        self.net = Net(softmax=softmax)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        #cache of (state tensor, label tensor, iteration) tuples
        #will eventually be put in training db
        self.sampleCacheSize = sampleCacheSize
        self.sampleCache = []


        #we need to store numpy arrays in sqlite
        #so these functions convert between the two
        #https://stackoverflow.com/questions/18621513/python-insert-numpy-array-into-sqlite3-database
        #and I converted it to postgresql with this
        #https://stackoverflow.com/questions/10529351/using-a-psycopg2-converter-to-retrieve-bytea-data-from-postgresql
        def numpy_to_binary(arr):
            out = io.BytesIO()
            np.save(out, arr)
            out.seek(0)
            #return sqlite3.Binary(out.read())
            return psycopg2.Binary(out.read())

        def binary_to_numpy(data, cur):
            buf = psycopg2.BINARY(data, cur)
            out = io.BytesIO(buf)
            out.seek(0)
            return np.load(out)



        #sqlite.register_adapter(np.ndarray, adapt_array)
        #sqlite.register_converter('ARRAY', convert_array)
        psycopg2.extensions.register_adapter(np.ndarray, numpy_to_binary)
        ARRAY = psycopg2.extensions.new_type(psycopg2.BINARY.values, 'ARRAY', binary_to_numpy)
        psycopg2.extensions.register_type(ARRAY)

        #self.trainingDb = sqlite3.connect(name + '.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.trainingDb = psycopg2.connect(dbConnect)
        self.cur = self.trainingDb.cursor()
        if clearDb:
            self.cur.execute('DROP TABLE IF EXISTS samples;')
            self.cur.execute('DROP SEQUENCE IF EXISTS sample_seq;')
        self.cur.execute('CREATE TABLE IF NOT EXISTS samples (id SERIAL PRIMARY KEY, data BYTEA);')
        self.cur.execute('CREATE SEQUENCE IF NOT EXISTS sample_seq')
        self.cur.execute('ALTER SEQUENCE sample_seq OWNED BY samples.id')
        self.trainingDb.commit()

    def addSample(self, data, label, iter):
        stateTensor = modelInput.stateToTensor(data)

        labelTensor = np.zeros(modelInput.numActions)
        for action, value in label:
            n = modelInput.enumAction(action)
            labelTensor[n] = value

        #put the np array in a tuple because that's what sqlite expects
        self.sampleCache.append((np.concatenate((stateTensor, labelTensor, [iter])),))
        if len(self.sampleCache) > self.sampleCacheSize:
            self.clearSampleCache()

    #moves all samples from cache to the db
    def clearSampleCache(self):
        if len(self.sampleCache) == 0:
            return
        self.cur.executemany(
                'INSERT INTO samples VALUES (nextval(\'sample_seq\'), %s)', self.sampleCache)
                #'INSERT INTO samples VALUES (NULL, ?)', [(numpy_to_bytea(s[0]),) for s in self.sampleCache])
        self.trainingDb.commit()
        self.sampleCache = []

    #we need to clean our db, clear out caches
    def close(self):
        #make sure we save everything first
        #so we can use the same training data in the future
        self.clearSampleCache()
        self.cur.close()
        self.trainingDb.close()

    def predict(self, state):
        data = modelInput.stateToTensor(state)
        data = torch.from_numpy(data).float()
        return self.net(data).detach().numpy()

    def train(self, epochs=100):

        #move from write cache to db
        self.clearSampleCache()

        #this is where we would send the model to the GPU for training
        #but my GPU is too old for that

        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        self.net = Net(softmax=self.softmax)
        #self.net.to(device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        miniBatchSize = 1000

        #can't train without any samples
        dataSetSize = self.getTrainingDbSize()
        if dataSetSize == 0:
            return
        print('dataset size:', dataSetSize, file=sys.stderr)

        for i in range(epochs):
            samples = self.getRandomSamples(miniBatchSize)
            #each row in samples is a sample, so we're getting the columns
            sampleData = samples[:, 0:modelInput.stateSize]
            sampleLabels = samples[:, modelInput.stateSize:modelInput.stateSize + modelInput.numActions]
            sampleIters = samples[:, -1]

            #convert each to a torch tensor
            data = torch.from_numpy(sampleData).float()
            #data.to(device)

            labels = torch.from_numpy(sampleLabels).float()
            #labels.to(device)

            iters = torch.from_numpy(sampleIters).float()
            #iters.to(device)

            #evaluate on network
            self.optimizer.zero_grad()
            ys = self.net(data)

            #loss function from the paper
            loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))
            #print the last 10 losses
            if i > epochs-11:
                print(loss, file=sys.stderr)
            #get gradient of loss
            loss.backward()
            #clip gradient norm, which was done in the paper
            nn.utils.clip_grad_norm_(self.net.parameters(), 1000)
            #train the network
            self.optimizer.step()

    def getTrainingDbSize(self):
        self.cur.execute('SELECT COUNT(*) FROM samples')
        size = self.cur.fetchone()[0]
        return size

    def getRandomSamples(self, num):
        self.cur.execute('SELECT data FROM samples ORDER BY RANDOM() LIMIT %s', (num,))
        #samples = self.cur.execute('SELECT data FROM samples ORDER BY RANDOM() LIMIT %s', (num,))
        #samples = [bytea_to_numpy(s[0]) for s in cur.fetchall()]
        samples = [s[0] for s in self.cur.fetchall()]
        return np.array(samples)
