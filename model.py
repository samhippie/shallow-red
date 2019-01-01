#!/usr/bin/env python3

from apex import amp
from hashembed.embedding import HashEmbedding
import io
import numpy as np
import sys
import torch
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
import torch.optim as optim
import torch.utils.data
from torchvision import transforms

import dataStorage
import config

#this should stop some errors about too many files being open
torch.multiprocessing.set_sharing_strategy('file_system')

amp_handle = amp.init()

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
    return torch.tensor(b)

#used to map each token in an infoset into an int representation
#the first number is used for embedding, and the other numbers are for binary numbers
numberMap = {}
def tokenToTensor(x):
    if not numberMap:
        for i in range(1 << NUM_TOKEN_BITS):
            numberMap[str(i)] = numToBinary(i)

    if x in numberMap:
        return torch.tensor([hash(x), *numberMap[x]])
    else:
        return torch.tensor([hash(x), *numberMap['0']])

#formats the infoset so it can be processed by the network
#this is how infosets should be stored
def infosetToTensor(infoset):
    #if type(infoset[0]) == list:
        #return [torch.stack([tokenToTensor(token) for token in seq]) for seq in infoset]
    return torch.stack([tokenToTensor(token) for token in infoset])

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
        self.embeddings = HashEmbedding(config.vocabSize, self.embedSize, append_weight=False, mask_zero=True)
        #LSTM to process infoset via the embeddings
        self.lstm = nn.LSTM(self.embedSize + NUM_TOKEN_BITS, config.lstmSize, batch_first=True)

        #simple feed forward for final part
        self.fc1 = nn.Linear(config.lstmSize, config.width)
        self.fc2 = nn.Linear(config.width, config.width)
        self.fc3 = nn.Linear(config.width, config.width)
        self.fc6 = nn.Linear(config.width, self.outputSize)

        #I don't know how this function works but whatever
        #that's how we roll
        #self.normalizer = nn.LayerNorm((width,))

    def forward(self, infoset, lengths=None):
        #I'm trying to be pretty aggressive about deleting things
        #as it's easy to run out of gpu memory with large sequences

        isSingle = False
        if len(infoset.shape) == 2:
            isSingle = True
            infoset = infoset[None, :, :]

        #embed the word hash, which is the first element
        #print('infoset', infoset)
        embedded = self.embeddings(infoset[:,:,0])
        #print('embedded', embedded)

        #embedding seems to spit out some pretty low-magnitude vectors
        #so let's try normalizing
        #embedded = F.normalize(embedded, p=2, dim=2)
        #replace the hash with the embedded vector
        embedded = torch.cat((embedded, infoset[:,:, 1:].float()), 2)
        del infoset

        #lengths are passed in if we have to worry about padding
        #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        if lengths is not None:
            embedded = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)

        #remember that we set batch_first to be true
        x, _ = self.lstm(embedded)

        #have to undo our packing before moving on
        if lengths is not None:
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            #select the last output before padding
            x = x[torch.arange(0, x.shape[0]), lengths-1]
        else:
            x = x[-1]

        #print('lstm output', x)
        del embedded

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc6(x)
        #print('out of linear', x)
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
        self.net.float()
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
        self.net = self.net.to(device)
        data = data.to(device)
        return self.net(data).cpu().detach().numpy()

    def train(self, epochs=1):
        #move from write cache to db
        self.clearSampleCache()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        self.net = Net(softmax=self.softmax)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.net = self.net.to(device)
        miniBatchSize = config.miniBatchSize

        def myCollate(batch):
            #based on the collate_fn here
            #https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py

            #sort by data length
            batch.sort(key=lambda x: len(x[0]), reverse=True)
            data, labels, iters = zip(*batch)

            #labels and iters have a fixed size, so we can just stack
            labels = torch.stack(labels)
            iters = torch.stack(iters)

            #sequences are padded with 0 vectors to make the lengths the same
            lengths = [len(d) for d in data]
            padded = torch.zeros(len(data), max(lengths), len(data[0][0]), dtype=torch.long)
            for i, d in enumerate(data):
                end = lengths[i]
                padded[i, :end] = d[:end]

            #need to know the lengths so we can pack later
            lengths = torch.tensor(lengths)

            return padded, lengths, labels, iters

        for j in range(epochs):
            if epochs > 1:
                print('epoch', j, file=sys.stderr)
            dataset = dataStorage.Dataset(self.name, self.sharedDict, self.outputSize)
            loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=True, num_workers=config.numWorkers, pin_memory=True, collate_fn=myCollate)

            print('dataset size:', dataset.size, file=sys.stderr)

            #print out loss every 1/10th of an epoch
            i = 0
            chunkSize = dataset.size  / (miniBatchSize * 10)
            for data, dataLengths, labels, iters in loader:
                i += 1

                labels = labels.float().to(device)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)

                #evaluate on network
                self.optimizer.zero_grad()
                ys = self.net(data, lengths=dataLengths)

                #loss function from the paper
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))
                del ys

                if i > chunkSize:
                    print(loss, file=sys.stderr)
                    i = 0

                #get gradient of loss
                #use amp because nvidia said it's better
                with amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                del loss

                #clip gradient norm, which was done in the paper
                #nn.utils.clip_grad_norm_(self.net.parameters(), 1000)

                #train the network
                self.optimizer.step()


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
