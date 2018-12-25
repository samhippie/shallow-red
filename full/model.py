#!/usr/bin/env python3

from hashembed.embedding import HashEmbedding
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

#I'm getting cudnn errors if I leave it enabled
#but it works without
#torch.backends.cudnn.enabled=False

#TODO
#I'm going to try disabling the way numbers are handled until batching and cuda both work
#once we have those working, find a good way of representing numbers

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

def myHash(x, axis):
    return hash(x)
vecHash = np.vectorize(myHash)
def infosetToTensor(infoset):
    if type(infoset[0]) == list:
        return [[hash(token) for token in seq] for seq in infoset]
    return [hash(token) for token in infoset]
    #if not numberMap:
        #for i in range(1024):
            #numberMap[str(i)] = numToBinary(i)
    #t = np.stack([np.concatenate([[1], numberMap[token]]) if token in numberMap else np.concatenate([[0, hash(token)], np.zeros(9, dtype=np.long)]) for token in infoset])
    #return t
    #return np.apply_over_axes(vecHash, np.array(infoset), (1,))

#model of the network
class Net(nn.Module):
    def __init__(self, embedSize=30, lstmSize=100, width=100, softmax=False):
        super(Net, self).__init__()

        self.outputSize = full.game.numActions

        self.softmax = softmax
        self.embedSize = embedSize
        #used for appending number representation to the embedding
        numSize = 0

        #turn vocab indices from history into embedding vectors
        #self.embeddings = nn.Embedding(self.vocabSize, embedSize)
        self.embeddings = HashEmbedding(10000, embedSize, append_weight=False)
        #LSTM to process infoset via the embeddings
        self.lstm = nn.LSTM(embedSize + numSize, lstmSize)

        #simple feed forward for final part
        self.fc1 = nn.Linear(lstmSize, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, width)
        self.fc6 = nn.Linear(width, self.outputSize)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, infoset, device=None):
        #convert infoset into token_repr | int_repr
        #so we can handle both tokens and numbers in the same input
        #lstmInput = []
        #for token in infoset:
            #if token[0] == 1: #number input
                #val = torch.cat([torch.zeros(self.embedSize), token[1:].float()])
            #else:
                #val = torch.cat([self.embeddings(token[1].view(1, -1)).view(-1), torch.zeros(10)])
            #lstmInput.append(val)
        #lstmInput = torch.stack(lstmInput)

        #make it BxLxD if it's just LxD
        isSingle = False
        if type(infoset[0]) == int:
            isSingle = True
            infoset = [infoset]

        lengths = torch.LongTensor([len(seq) for seq in infoset])
        if device:
            lengths = lengths.to(device)
        seq_tensor = torch.zeros((len(infoset), lengths.max())).long()
        if device:
            seq_tensor = seq_tensor.to(device)
            for idx, (seq, seqlen) in enumerate(zip(infoset, lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq).to(device)
        else:
            for idx, (seq, seqlen) in enumerate(zip(infoset, lengths)):
                seq_tensor[idx, :seqlen] = torch.LongTensor(seq)

        #sort by sequence length
        lengths, sort_idx = lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[sort_idx]
        #LSTM expects a shape of (L, B, D)
        #?
        #infoset = infoset.transpose(0, 1)

        seq_tensor = torch.transpose(seq_tensor, 0, 1)
        embedded = self.embeddings(seq_tensor)
        #LSTM expects a shape of (L, B, D)
        #embedded = torch.transpose(embedded, 0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths)

        #_, lstmOutput = self.lstm(lstmInput.view(len(infoset), 1, -1), self.hidden)
        #_, (ht, ct) = self.lstm(packed, self.hidden)
        _, (ht, ct) = self.lstm(packed)
        #print('output shape', output.shape)
        #print('hidden shape', hidden.shape)
        #output, hidden = torch.nn.utils.rnn.pad_packed_sequence(output)
        #print('output2 shape', output.shape)
        #print('hidden2 shape', hidden.shape)
        #lstmOutput = torch.transpose(hidden, 0, 1)
        lstmOutput = ht[-1]

        #restore original order in batch
        lstmOutput = lstmOutput[sort_idx.sort(0)[1]]

        #print('lstm output', ht)
        #lstmOutput = lstmOutput[0].view(-1)
        x = F.relu(self.fc1(lstmOutput))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc6(x)
        #normalize to 0 mean and unit variance
        #like in the paper
        #x = self.normalizer(x)
        if self.softmax:
            x = F.softmax(x, dim=0)

        if isSingle:
            return x[0]
        else:
            return x

    #pytorch doesn't know to move the LSTM default hidden input
    #I don't really like having a side effect in this sort of function but whatever
    #def moveToDevice(self, device):
        #self.hidden = tuple(h.to(device) for h in self.hidden)
        #return self.to(device)

class DeepCfrModel:

    #for advantages, the input is the state vector
    #and the output is a vector of each move's advantage
    #for strategies, the input is the state vector
    #and the output is a vector of each move's probability

    #so the inputs are exactly the same (modelInput.stateSize), and the outputs
    #are almost the same (modelInput.numActions)
    #strategy is softmaxed, advantage is not

    def __init__(self, name, softmax, writeLock, sharedDict, lr=0.0001, sampleCacheSize=1000, clearDb=True):
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
        #data = torch.from_numpy(data).long()
        return self.net(data).detach().numpy()

    def train(self, epochs=100):
        #move from write cache to db
        self.clearSampleCache()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        self.net = Net(softmax=self.softmax)
        self.net = self.net.to(device)
        #self.net = self.net.moveToDevice(device)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        miniBatchSize = 4

        #data needs to be a python list of python lists due to variable lengths
        #everything else can be a proper tensor
        def my_collate(batch):
            data = [b[0] for b in batch]
            labels = torch.stack([b[1] for b in batch])
            iters = torch.stack([b[2] for b in batch])
            return data, labels, iters

        dataset = full.dataStorage.Dataset(self.name, self.sharedDict, self.outputSize)
        loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=True, num_workers=4, pin_memory=False, collate_fn=my_collate)

        print('dataset size:', dataset.size, file=sys.stderr)

        batchIter = iter(loader)
        for i in range(epochs):
            #print('getting data from loader', file=sys.stderr)
            try:
                data, labels, iters = next(batchIter)
            except StopIteration:
                batchIter = iter(loader)
                data, labels, iters = next(batchIter)

            labels = labels.float()
            iters = iters.float()
            #print('moving data to device', file=sys.stderr)
            #data = data.to(device)
            labels = labels.to(device)
            iters = iters.to(device)

            #print('getting ys', file=sys.stderr)
            #evaluate on network
            self.optimizer.zero_grad()
            ys = self.net(data, device)

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
        #self.net = self.net.moveToDevice(torch.device('cpu'))

def netTest():
    net = Net()
    #simple test
    out1 = net.forward([1,2,3])
    out2 = net.forward([4,5,6])
    print('simple1', out1)
    print('simple2', out2)
    #batch test
    out = net.forward([
        [1,2,3],
        [4,5,6],
        [7,8,9],
    ])
    print('batch', out)
    #multi length batch test
    out = net.forward([
        [4,5],
        [1,2,3],
        [4,5],
    ])
    print('multi length batch', out)
