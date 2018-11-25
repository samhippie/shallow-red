#!/usr/bin/env python3

import numpy as np
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms

import modelInput

#model of the network
class Net(nn.Module):
    def __init__(self, softmax=False, width=200):
        super(Net, self).__init__()

        self.softmax = softmax

        #simple feed forward
        self.fc1 = nn.Linear(modelInput.stateSize, width)
        self.fc2 = nn.Linear(width, width)
        #self.fc3 = nn.Linear(width, width)
        #self.fc4 = nn.Linear(width, width)
        #self.fc5 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, modelInput.numActions)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x))
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


    def __init__(self, softmax, lr=0.001):
        self.dataSet = []
        self.labelSet = []
        self.iterSet = []

        self.softmax = softmax
        self.lr = lr

        self.net = Net(softmax=softmax)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

    def addSample(self, data, label, iter):
        self.dataSet.append(modelInput.stateToTensor(data))

        labelDense = np.zeros(modelInput.numActions)
        for action, value in label:
            n = modelInput.enumAction(action)
            labelDense[n] = value
        self.labelSet.append(labelDense)

        self.iterSet.append(iter)

    def predict(self, state):
        data = modelInput.stateToTensor(state)
        data = torch.from_numpy(data).float()
        return self.net(data).detach().numpy()

    def train(self, epochs=100):

        #this is where we would send the model to the GPU for training
        #but my GPU is too old for that

        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

        self.net = Net(softmax=self.softmax)
        self.net.to(device)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        miniBatchSize = 100

        #can't train without any samples
        if len(self.dataSet) == 0:
            return
        print('dataset size:', len(self.dataSet), file=sys.stderr)

        dataSet = np.array(self.dataSet)
        labelSet = np.array(self.labelSet)
        iterSet = np.array(self.iterSet)
        for i in range(epochs):
            sampleIndices = np.random.choice(len(dataSet), miniBatchSize)

            sampleData = dataSet[sampleIndices]
            data = torch.from_numpy(sampleData).float()
            data.to(device)

            sampleLabels = labelSet[sampleIndices]
            labels = torch.from_numpy(sampleLabels).float()
            labels.to(device)

            sampleIters = iterSet[sampleIndices]
            iters = torch.from_numpy(sampleIters).float()
            iters.to(device)

            self.optimizer.zero_grad()
            ys = self.net(data)

            #loss function from the paper
            loss = torch.sum(iters.view(miniBatchSize,1) * ((labels - ys) ** 2))
            #print the last 10 losses
            if i > epochs-11:
                print(loss, file=sys.stderr)
            loss.backward()
            #clip gradient norm, which was done in the paper
            nn.utils.clip_grad_norm_(self.net.parameters(), 1000)
            self.optimizer.step()

