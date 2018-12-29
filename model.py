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

import dataStorage
import config

#how many bits are used to represent numbers in tokens
NUM_TOKEN_BITS = config.numTokenBits

def numToBinary(n):
    ceiling = 1 << NUM_TOKEN_BITS
    if n >= ceiling:
        n = ceiling - 1
    b = []
    while n > 0:
        b.append(n & 1)
        n >>= 1
    b += [0] * (NUM_TOKEN_BITS - len(b))
    return np.array(b)

#used to map each token in an infoset into an int representation
#the first number is used for embedding, and the other numbers are for binary numbers
numberMap = {}
def tokenToTensor(x):
    if not numberMap:
        for i in range(1 << NUM_TOKEN_BITS):
            numberMap[str(i)] = numToBinary(i)

    if x in numberMap:
        return [0, *numberMap[x]]
    else:
        return [hash(x), *numberMap['0']]

#formats the infoset so it can be processed by the network
#this is how infosets should be stored
def infosetToTensor(infoset):
    if type(infoset[0]) == list:
        return [[tokenToTensor(token) for token in seq] for seq in infoset]
    return [tokenToTensor(token) for token in infoset]

#model of the network
class Net(nn.Module):
    #embed size is how large our embedding output is
    #lstmSize is the size of the lstm hidden output
    #width is the size of the feedforward layers
    #softmax is whether we softmax the final output
    def __init__(self, softmax=False):
        super(Net, self).__init__()

        self.outputSize = config.game.numActions

        self.softmax = softmax
        self.embedSize = config.embedSize

        #turn vocab indices from history into embedding vectors
        #self.embeddings = nn.Embedding(self.vocabSize, embedSize)
        self.embeddings = HashEmbedding(10000, self.embedSize, append_weight=False)
        #LSTM to process infoset via the embeddings
        self.lstm = nn.LSTM(self.embedSize + NUM_TOKEN_BITS, config.lstmSize)

        #simple feed forward for final part
        self.fc1 = nn.Linear(config.lstmSize, config.width)
        self.fc2 = nn.Linear(config.width, config.width)
        self.fc3 = nn.Linear(config.width, config.width)
        self.fc6 = nn.Linear(config.width, self.outputSize)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, infoset, device=None):
        #about the infoset input
        #it should be a batch, which is a set of sequences
        #and a sequence is a series of tokens
        #and each token is a [hash, bit, bit, ...]
        #batch -> sequence -> token -> hash and bits


        #all the commented out stuff is staying because I'm not sure everything is lined up correctly
        #after all the transposing and sorting and whatever else to make the lstm happy
        #this part is based on this
        #https://gist.github.com/Tushar-N/dfca335e370a2bc3bc79876e6270099e

        #make it a batch of 1 in case it's just a single
        isSingle = False
        if type(infoset[0][0]) == int:
            isSingle = True
            infoset = [infoset]

        lengths = torch.LongTensor([len(seq) for seq in infoset])
        if device:
            lengths = lengths.to(device)
        seq_tensor = torch.zeros((len(infoset), lengths.max(), NUM_TOKEN_BITS + 1)).long()
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
        #keep the batches and sequence, but only pick the first element of the token i.e. the hash
        embedded = self.embeddings(seq_tensor[:, :, 0])
        #replace the hash with the embedded vector
        embedded = torch.cat((embedded, seq_tensor[:, :, 1:].float()), 2)


        #LSTM expects a shape of (L, B, D)
        #embedded = torch.transpose(embedded, 0, 1)

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths)
        del lengths
        del seq_tensor

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

        #lstmOutput = lstmOutput[0].view(-1)
        x = F.relu(self.fc1(lstmOutput))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc6(x)
        #normalize to 0 mean and unit variance
        #like in the paper
        #x = self.normalizer(x)
        if self.softmax:
            x = F.softmax(x, dim=1)

        if isSingle:
            return x[0]
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

    def __init__(self, name, softmax, writeLock, sharedDict):
        self.softmax = softmax
        self.lr = config.learnRate
        self.writeLock = writeLock
        self.sharedDict = sharedDict
        self.outputSize = config.game.numActions

        self.net = Net(softmax=softmax)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)

        #cache of (infoset tensor, label tensor, iteration) tuples
        #will eventually be put in training db
        self.sampleCacheSize = config.sampleCacheSize
        self.sampleCache = []

        self.name = name

    def shareMemory(self):
        self.net.share_memory()

    def addSample(self, infoset, label, iter):
        #infosetTensor = np.array([hash(token) for token in infoset], dtype=np.long)
        infosetTensor = infosetToTensor(infoset)

        labelTensor = np.zeros(self.outputSize)
        for action, value in label:
            n = config.game.enumAction(action)
            labelTensor[n] = value

        iterTensor = np.array([iter])

        self.sampleCache.append((infosetTensor, labelTensor, iterTensor))
        if len(self.sampleCache) > self.sampleCacheSize:
            self.clearSampleCache()

    #moves all samples from cache to the db
    def clearSampleCache(self):
        if len(self.sampleCache) == 0:
            return
        dataStorage.addSamples(self.writeLock, self.name, self.sampleCache, self.sharedDict)
        self.sampleCache = []

    #we need to clean our db, clear out caches
    def close(self):
        #make sure we save everything first
        #so we can use the same training data in the future
        self.clearSampleCache()

    def predict(self, infoset):
        data = infosetToTensor(infoset)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net.to(device)
        return self.net(data, device).cpu().detach().numpy()

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
        miniBatchSize = 32

        #data needs to be a python list of python lists due to variable lengths
        #everything else can be a proper tensor
        def my_collate(batch):
            data = [b[0] for b in batch]
            labels = torch.stack([b[1] for b in batch])
            iters = torch.stack([b[2] for b in batch])
            return data, labels, iters

        dataset = dataStorage.Dataset(self.name, self.sharedDict, self.outputSize)
        loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=True, num_workers=4, pin_memory=True, collate_fn=my_collate)

        print('dataset size:', dataset.size, file=sys.stderr)

        batchIter = iter(loader)
        for i in range(epochs):
            #print('getting data from loader', file=sys.stderr)
            try:
                data, labels, iters = next(batchIter)
            except StopIteration:
                batchIter = iter(loader)
                data, labels, iters = next(batchIter)

            labels = labels.float().to(device)
            iters = iters.float().to(device)

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

            #cleaning up might improve memory usage
            del loss
            del ys

        #self.net = self.net.to(torch.device('cpu'))

def netTest():
    net = Net()
    list1 = infosetToTensor(['a', 'b', 'c'])
    list2 = infosetToTensor(['d', 'e', 'f'])
    list3 = infosetToTensor(['g', 'h'])
    #simple test
    out1 = net.forward(list1)
    out2 = net.forward(list2)
    print('simple1', out1)
    print('simple2', out2)
    #batch test
    out = net.forward([
        list1,
        list2,
    ])
    print('batch', out)
    #multi length batch test
    out = net.forward([
        list3,
        list1,
        list3,
    ])
    print('multi length batch', out)
