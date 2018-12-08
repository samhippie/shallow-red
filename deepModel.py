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
#the topology of this really should be configurable
class Net(nn.Module):
    def __init__(self, softmax=False, width=300):
        super(Net, self).__init__()

        self.softmax = softmax

        #simple feed forward
        #this is kind of big, but I waste CPU cycles
        #with smaller networks (given mini-batching)
        #and this should be even more true with a GPU
        self.fc1 = nn.Linear(modelInput.stateSize, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        #self.fc4 = nn.Linear(width, width)
        #self.fc5 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, modelInput.numActions)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        #x = F.relu(self.fc4(x))
        #x = F.relu(self.fc5(x))
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

        #if we're not clearing the db, then we should also load in the id map
        #so that the inputs to the model will match those in the db
        if not clearDb:
            modelInput.readIdMap('idmap.pickle')

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

        #we're going to be inserting this directly into the query strings
        #because psql doesn't like table names from parameters
        self.tableName = 'samples_' + name

        #self.trainingDb = sqlite3.connect(name + '.db', detect_types=sqlite3.PARSE_DECLTYPES)
        self.trainingDb = psycopg2.connect(dbConnect)
        cur = self.trainingDb.cursor()
        if clearDb:
            cur.execute('DROP TABLE IF EXISTS ' + self.tableName)
            cur.execute('DROP SEQUENCE IF EXISTS ' + self.tableName + '_seq')
        #cur.execute('CREATE TABLE IF NOT EXISTS samples (id SERIAL PRIMARY KEY, data BYTEA);')
        cur.execute('CREATE TABLE IF NOT EXISTS ' + self.tableName + ' (id SERIAL PRIMARY KEY, data BYTEA);')
        cur.execute('CREATE SEQUENCE IF NOT EXISTS ' + self.tableName + '_seq')
        cur.execute('ALTER SEQUENCE ' + self.tableName + '_seq OWNED BY ' + self.tableName + '.id')
        cur.close()
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
        cur = self.trainingDb.cursor()
        cur.executemany(
                'INSERT INTO ' + self.tableName + ' VALUES (nextval(\'sample_seq\'), %s)', self.sampleCache)
                #'INSERT INTO samples VALUES (NULL, ?)', [(numpy_to_bytea(s[0]),) for s in self.sampleCache])
        cur.close()
        self.trainingDb.commit()
        self.sampleCache = []

    #we need to clean our db, clear out caches
    def close(self):
        #make sure we save everything first
        #so we can use the same training data in the future
        self.clearSampleCache()
        self.trainingDb.close()

    def predict(self, state):
        data = modelInput.stateToTensor(state)
        data = torch.from_numpy(data).float()
        return self.net(data).detach().numpy()

    def train(self, epochs=100):
        #TODO make this configurable
        #I'm doing this so we can manually resume a stopped run
        modelInput.saveIdMap('idmap.pickle')

        #move from write cache to db
        self.clearSampleCache()

        #this is where we would send the model to the GPU for training
        #but my GPU is too old for that

        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        print('initing net')
        self.net = Net(softmax=self.softmax)
        #self.net.to(device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        miniBatchSize = 1000

        print('getting data size')
        #can't train without any samples
        dataSetSize = self.getTrainingDbSize()
        if dataSetSize == 0:
            return
        print('dataset size:', dataSetSize, file=sys.stderr)

        print('getting all samples')
        sampleSet = self.getRandomSamples(miniBatchSize, dataSetSize, epochs)

        for i in range(epochs):
            #print('getting samples')
            #samples = self.getRandomSamples(miniBatchSize, dataSetSize)
            samples = sampleSet[i]
            print('slicing/converting samples')
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

            print('evaluating samples')

            #evaluate on network
            self.optimizer.zero_grad()
            ys = self.net(data)

            print('getting loss')

            #loss function from the paper
            loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))
            #print the last 10 losses
            if i > epochs-11:
                print(loss, file=sys.stderr)
            print('backprop')
            #get gradient of loss
            loss.backward()
            print('clipping gradient')
            #clip gradient norm, which was done in the paper
            nn.utils.clip_grad_norm_(self.net.parameters(), 1000)
            print('optimizing net')
            #train the network
            self.optimizer.step()
            print('done with epoch', i)

    def getTrainingDbSize(self):
        cur = self.trainingDb.cursor()
        cur.execute('SELECT COUNT(*) FROM ' + self.tableName)
        size = cur.fetchone()[0]
        cur.close()
        return size

    def getRandomSamples(self, batchSize, totalSize, numBatches):
        cur = self.trainingDb.cursor()
        #this is supposed to be fast
        percent = min(100 * batchSize / totalSize, 100)
        print('getting percentage', percent)
        #our samples shouldn't be that big, so the extra randomness from bernoulli should be worth the speed tradeoff
        #cur.execute('SELECT data FROM ' + self.tableName + ' TABLESAMPLE SYSTEM (%s)', (percent,))
        cur.executemany('SELECT data FROM ' + self.tableName + ' TABLESAMPLE SYSTEM (%s)', ((percent,),) * numBatches)
        #naive way, but works
        #cur.execute('SELECT data FROM ' + self.tableName + ' ORDER BY RANDOM() LIMIT %s', (num,))
        #don't remember what these are from, probably don't work
        #samples = self.cur.execute('SELECT data FROM samples ORDER BY RANDOM() LIMIT %s', (num,))
        #samples = [bytea_to_numpy(s[0]) for s in cur.fetchall()]
        samples = [s[0][0] for s in cur.fetchall()]
        cur.close()
        return np.array(samples)
