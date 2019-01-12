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
        self.embeddings = HashEmbedding(config.vocabSize, self.embedSize, append_weight=False, mask_zero=False)

        self.dropout = nn.Dropout(config.embedDropoutPercent)

        #LSTM to process infoset via the embeddings
        self.lstm = nn.LSTM(self.embedSize + NUM_TOKEN_BITS, config.lstmSize, num_layers=config.numLstmLayers, batch_first=True)

        #attention
        self.attn = nn.Linear(2 * config.lstmSize, config.lstmSize)

        #simple feed forward for final part
        self.fc1 = nn.Linear(config.lstmSize, config.width)
        self.fc2 = nn.Linear(config.width, config.width)
        self.fc3 = nn.Linear(config.width, config.width)
        self.fc6 = nn.Linear(config.width, self.outputSize)

    def forward(self, infoset, lengths=None, trace=False):
        #I'm trying to be pretty aggressive about deleting things
        #as it's easy to run out of gpu memory with large sequences

        isSingle = False
        if len(infoset.shape) == 2:
            isSingle = True
            infoset = infoset[None, :, :]

        #embed the word hash, which is the first element
        if trace:
            print('infoset', infoset, file=sys.stderr)
        embedded = self.embeddings(infoset[:,:,0])
        if trace:
            print('embedded', embedded, file=sys.stderr)

        #embedding seems to spit out some pretty low-magnitude vectors
        #so let's try normalizing
        embedded = F.normalize(embedded, p=2, dim=2)
        embedded = self.dropout(embedded)
        #replace the hash with the embedded vector
        embedded = torch.cat((embedded, infoset[:,:, 1:].float()), 2)
        del infoset

        #lengths are passed in if we have to worry about padding
        #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        if lengths is not None:
            lstmInput = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        else:
            lstmInput = embedded

        #remember that we set batch_first to be true
        x, _ = self.lstm(lstmInput)

        #get the final output of the lstm
        if lengths is not None:
            #have to account for padding/packing
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            lasts = x[torch.arange(0, x.shape[0]), lengths-1]
        else:
            lasts = x[:,-1]

        if trace:
            print('lasts', lasts, file=sys.stderr)
        #use both input and output of lstm to get attention weights
        #keep the batch size, but add extra dimension for sequence length
        lasts = lasts[:, None, :]
        #repeat the last output so it matches the sequence length
        lasts = lasts.repeat(1, x.shape[1], 1)
        #feed the output of the lstm appended with the final output to the attention layer
        xWithContext = torch.cat([x, lasts], 2)
        outattn = self.attn(xWithContext)
        #softmax so the weights for each element of each output add up to 1
        outattn = F.softmax(outattn, dim=1)
        #apply the weights to each output
        #x and outattn are the same shape, so this is easy
        #if we padded, then the padding outputs in x are 0, so they don't contribute to the sum
        #so it just works
        x = x * outattn
        #sum along the sequence
        x = torch.sum(x, dim=1)
        if trace:
            print('x to fc', x, file=sys.stderr)

        x = F.relu(self.fc1(x))
        #2 and 3 have skip connections, as they have the same sized input and output
        x = F.relu(self.fc2(x) + x)
        x = F.relu(self.fc3(x) + x)
        #deep cfr does normalization here
        #I'm not sure if this is the right kind of normalization
        #x = F.normalize(x, p=2, dim=1)
        x = self.fc6(x)

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
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True)

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
            #print(action, value)
            n = config.game.enumAction(action)
            labelTensor[n] = value
        #print('saving label', labelTensor)

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

    def predict(self, infoset, trace=False):
        data = infosetToTensor(infoset)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(device)
        data = data.to(device)
        return self.net(data, trace=trace).cpu().detach().numpy()

    def train(self, epochs=1):
        #move from write cache to db
        self.clearSampleCache()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        if config.newIterNets:
            self.net = Net(softmax=self.softmax)


        self.net = self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, verbose=True)
        miniBatchSize = config.miniBatchSize
        self.net.train(True)

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

            totalLoss = 0

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
                ys = self.net(data, lengths=dataLengths, trace=False)
                #print('------------', file=sys.stderr)
                #print('ys', ys, file=sys.stderr)
                #print('labels', labels, file=sys.stderr)
                #print('iters', iters, file=sys.stderr)
                #print('scaled loss', iters.view(labels.shape[0],-1) * ((labels - ys) ** 2), file=sys.stderr)

                #loss function from the paper
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))
                del ys

                #get gradient of loss
                #use amp because nvidia said it's better
                with amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()


                #clip gradient norm, which was done in the paper
                nn.utils.clip_grad_norm_(self.net.parameters(), 1000)

                #train the network
                self.optimizer.step()
                totalLoss += loss.item()

            avgLoss = totalLoss / dataset.size
            #self.scheduler.step(avgLoss)
            print('avgLoss', avgLoss, file=sys.stderr)

        self.net.train(False)

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
