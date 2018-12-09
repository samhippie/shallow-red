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
import torch.utils.data
from torchvision import transforms

import modelInput
import deep.dataStorage

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

    def __init__(self, name, softmax, writeLock, sharedDict, lr=0.001, sampleCacheSize=1000, clearDb=True):
        self.softmax = softmax
        self.lr = lr
        self.writeLock = writeLock
        self.sharedDict = sharedDict

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

        self.name = name

    def addSample(self, data, label, iter):
        stateTensor = modelInput.stateToTensor(data)

        labelTensor = np.zeros(modelInput.numActions)
        for action, value in label:
            n = modelInput.enumAction(action)
            labelTensor[n] = value

        #put the np array in a tuple because that's what sqlite expects
        self.sampleCache.append(np.concatenate((stateTensor, labelTensor, [iter])))
        if len(self.sampleCache) > self.sampleCacheSize:
            self.clearSampleCache()

    #moves all samples from cache to the db
    def clearSampleCache(self):
        if len(self.sampleCache) == 0:
            return
        deep.dataStorage.addSamples(self.writeLock, self.name, self.sampleCache, self.sharedDict)
        self.sampleCache = []

    #we need to clean our db, clear out caches
    def close(self):
        #make sure we save everything first
        #so we can use the same training data in the future
        self.clearSampleCache()

    def predict(self, state):
        data = modelInput.stateToTensor(state)
        data = torch.from_numpy(data).float()
        return self.net(data).detach().numpy()

    def train(self, epochs=100):
        #I'm doing this so we can manually resume a stopped run
        modelInput.saveIdMap('idmap.pickle')

        #move from write cache to db
        self.clearSampleCache()

        #this is where we would send the model to the GPU for training
        #but my GPU is too old for that

        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        self.net = Net(softmax=self.softmax)
        self.net.to(device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        miniBatchSize = 100

        dataset = deep.dataStorage.Dataset(self.name, self.sharedDict)
        loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=True, num_workers=4)

        print('dataset size:', dataset.size, file=sys.stderr)

        for i in range(epochs):
            for data, labels, iters in loader:
                #evaluate on network
                self.optimizer.zero_grad()
                ys = self.net(data)

                #loss function from the paper
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))
                #print the last 10 losses
                if i > epochs-11:
                    print(i, loss, file=sys.stderr)
                #get gradient of loss
                loss.backward()
                #clip gradient norm, which was done in the paper
                nn.utils.clip_grad_norm_(self.net.parameters(), 1000)
                #train the network
                self.optimizer.step()

        self.net.to(torch.device('cpu'))
