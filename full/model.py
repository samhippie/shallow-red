#!/usr/bin/env python3

from hashembed import HashEmbedding
import io
import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import transforms

import full.dataStorage
import full.game

def numToBinary(num):
    n = num % 1024
    b = []
    while n > 0:
        b.append(n & 1)
        n >>= 1
    b += [0] * (10 - len(b))
    return np.array(b)

#formats the infoset so it can be processed by the network
#this is how infosets should be stored
numberMap = {}
def infosetToTensor(infoset):
    if not numberMap:
        for i in range(1024):
            numberMap[str(i)] = numToBinary(i)
    t = np.stack([np.concatenate([[1], numberMap[token]]) if token in numberMap else np.concatenate([[0, hash(token)], np.zeros(9, dtype=np.long)]) for token in infoset])
    return t

#model of the network
class Net(nn.Module):
    def __init__(self, embedSize=20, lstmSize=30, width=30, softmax=False):
        super(Net, self).__init__()

        self.outputSize = full.game.numActions

        self.softmax = softmax
        self.embedSize = embedSize

        #turn vocab indices from history into embedding vectors
        #self.embeddings = nn.Embedding(self.vocabSize, embedSize)
        self.embeddings = HashEmbedding(10000, embedSize, append_weight=False)
        #LSTM to process infoset via the embeddings
        self.lstm = nn.LSTM(embedSize + 10, lstmSize)

        #lstm needs a default hidden tensor
        self.hidden = (torch.zeros(1, 1, lstmSize),
                torch.zeros(1, 1, lstmSize))

        #simple feed forward for final part
        self.fc1 = nn.Linear(lstmSize, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, self.outputSize)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, infoset):
        #convert infoset into token_repr | int_repr
        #so we can handle both tokens and numbers in the same input
        lstmInput = []
        for token in infoset:
            if token[0] == 1: #number input
                val = torch.cat([torch.zeros(self.embedSize), token[1:].float()])
            else:
                val = torch.cat([self.embeddings(token[1].view(1, -1)).view(-1), torch.zeros(10)])
            lstmInput.append(val)
        lstmInput = torch.stack(lstmInput)

        _, lstmOutput = self.lstm(lstmInput.view(len(infoset), 1, -1), self.hidden)
        lstmOutput = lstmOutput[0].view(-1)
        x = F.relu(self.fc1(lstmOutput))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc6(x)
        #normalize to 0 mean and unit variance
        #like in the paper
        #x = self.normalizer(x)
        if self.softmax:
            return F.softmax(x, dim=0)
        else:
            return x

class DeepCfrModel:

    #for advantages, the input is the state vector
    #and the output is a vector of each move's advantage
    #for strategies, the input is the state vector
    #and the output is a vector of each move's probability

    #so the inputs are exactly the same (modelInput.stateSize), and the outputs
    #are almost the same (modelInput.numActions)
    #strategy is softmaxed, advantage is not

    def __init__(self, name, softmax, writeLock, sharedDict, lr=0.0001, sampleCacheSize=10000, clearDb=True):
        self.softmax = softmax
        self.lr = lr
        self.writeLock = writeLock
        self.sharedDict = sharedDict
        self.outputSize = full.game.numActions

        self.net = Net(softmax=softmax)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        #cache of (infoset tensor, label tensor, iteration) tuples
        #will eventually be put in training db
        self.sampleCacheSize = sampleCacheSize
        self.sampleCache = []

        self.name = name

    def addSample(self, infoset, label, iter):
        #infosetTensor = np.array([hash(token) for token in infoset], dtype=np.long)
        infosetTensor = infosetToTensor(infoset)

        labelTensor = np.zeros(self.outputSize)
        for action, value in label:
            n = full.game.enumAction(action)
            labelTensor[n] = value

        iterTensor = np.array([iter])

        self.sampleCache.append((infosetTensor, labelTensor, iterTensor))
        if len(self.sampleCache) > self.sampleCacheSize:
            self.clearSampleCache()

    #moves all samples from cache to the db
    def clearSampleCache(self):
        if len(self.sampleCache) == 0:
            return
        full.dataStorage.addSamples(self.writeLock, self.name, self.sampleCache, self.sharedDict)
        self.sampleCache = []

    #we need to clean our db, clear out caches
    def close(self):
        #make sure we save everything first
        #so we can use the same training data in the future
        self.clearSampleCache()

    def predict(self, infoset):
        #data = np.array([hash(token) for token in infoset], dtype=np.long)
        data = infosetToTensor(infoset)
        data = torch.from_numpy(data).long()
        return self.net(data).detach().numpy()

    def train(self, epochs=100):
        #move from write cache to db
        self.clearSampleCache()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.net = Net(softmax=self.softmax)
        self.net = self.net.to(device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        miniBatchSize = 4

        dataset = full.dataStorage.Dataset(self.name, self.sharedDict, self.outputSize)
        loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=True, num_workers=4, pin_memory=True)

        print('dataset size:', dataset.size, file=sys.stderr)

        batchIter = iter(loader)
        for i in range(epochs):
            #print('getting data from loader', file=sys.stderr)
            try:
                data, labels, iters = next(batchIter)
            except StopIteration:
                batchIter = iter(loader)
                data, labels, iters = next(batchIter)

            #print('moving data to device', file=sys.stderr)
            data = data.to(device)
            labels = labels.to(device)
            iters = iters.to(device)
            
            #print('getting ys', file=sys.stderr)
            #evaluate on network
            self.optimizer.zero_grad()
            ys = self.net(data)

            #print('getting loss', file=sys.stderr)
            #loss function from the paper
            loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))
            #print the last 10 losses
            if i > epochs-11:
                print(i, loss, file=sys.stderr)
            #get gradient of loss
            #print('backward', file=sys.stderr)
            loss.backward()
            #clip gradient norm, which was done in the paper
            #print('clip', file=sys.stderr)
            nn.utils.clip_grad_norm_(self.net.parameters(), 1000)
            #train the network
            #print('step', file=sys.stderr)
            self.optimizer.step()
            #print('done with step', file=sys.stderr)

        self.net = self.net.to(torch.device('cpu'))

if __name__ == '__main__':
    print(numToBinary('255'))
