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
from torch.utils.data.sampler import SubsetRandomSampler

import dataStorage
import config

#this should stop some errors about too many files being open
#torch.multiprocessing.set_sharing_strategy('file_system')

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
        return torch.tensor([hash(x) % config.vocabSize, *numberMap[x]])
    else:
        return torch.tensor([hash(x) % config.vocabSize, *numberMap['0']])

#formats the infoset so it can be processed by the network
#this is how infosets should be stored
def infosetToTensor(infoset):
    return torch.stack([tokenToTensor(token) for token in infoset])

class SimpleNet(nn.Module):
    def __init__(self, softmax=False):
        super(Net, self).__init__()
        self.outputSize = config.game.numActions
        self.softmax = softmax
        self.inputSize = 15
        dropout = 0.5
        self.embeddings = HashEmbedding(config.vocabSize, config.embedSize, append_weight=False, mask_zero=True)
        self.dropE = nn.Dropout(dropout)

        #simple feed forward for final part
        self.fc1 = nn.Linear(self.inputSize * (config.embedSize + config.numTokenBits), config.width)
        self.drop1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(config.width, config.width)
        self.drop2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(config.width, config.width)
        self.drop3 = nn.Dropout(dropout)
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
        #embedded = F.normalize(embedded, p=2, dim=2)
        #embedded = self.dropout(embedded)
        #replace the hash with the embedded vector
        embedded = torch.cat((embedded, infoset[:,:, 1:].float()), 2)
        del infoset

        x = F.pad(embedded, (0,0, 0,self.inputSize - embedded.shape[1]))
        x = torch.cat(tuple(x.transpose(0,1)), 1)
        x = self.dropE(x)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        #2 and 3 have skip connections, as they have the same sized input and output
        x = F.relu(self.fc2(x) + x)
        x = self.drop2(x)
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x) + x)
        x = self.drop3(x)
        #x = F.relu(self.fc3(x))
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


#model of the network
class LstmNet(nn.Module):
    #softmax is whether we softmax the final output
    def __init__(self, softmax=False):
        super(Net, self).__init__()

        self.outputSize = config.game.numActions

        self.softmax = softmax
        self.embedSize = config.embedSize

        #turn vocab indices from history into embedding vectors
        self.embeddings = HashEmbedding(config.vocabSize, self.embedSize, append_weight=False, mask_zero=True)
        #self.embeddings = nn.Embedding(config.vocabSize, self.embedSize)

        self.dropout = nn.Dropout(config.embedDropoutPercent)

        """
        convLayers = []
        convBatchNorms = []
        for i, depth in enumerate(config.convDepths):
            convs = []
            convsBn = []
            for j in range(depth):
                if i == 0 and j == 0:
                    #input from embedding
                    inputSize = config.embedSize + config.numTokenBits
                elif j == 0:
                    #input from previous layer
                    inputSize = config.convSizes[i-1]
                else:
                    #input from current layer
                    inputSize = config.convSizes[i]
                #TODO make the stride configurable per layer so we can reduce the sequence length
                k = config.kernelSizes[i]
                p = k + 1 // 2
                conv = nn.Conv1d(inputSize, config.convSizes[i], kernel_size=k, stride=1, padding=p)
                convs.append(conv)
                convsBn.append(nn.BatchNorm1d(config.convSizes[i]))
            convLayers.append(nn.ModuleList(convs))
            convBatchNorms.append(nn.ModuleList(convsBn))
        self.convLayers = nn.ModuleList(convLayers)
        self.convBatchNorms = nn.ModuleList(convBatchNorms)
        """

        #LSTM to process infoset via the embeddings
        #self.lstm = nn.LSTM(config.convSizes[-1], config.lstmSize, num_layers=config.numLstmLayers, dropout=config.lstmDropoutPercent, batch_first=True)
        self.lstm = nn.LSTM(config.embedSize + config.numTokenBits, config.lstmSize, num_layers=config.numLstmLayers, dropout=config.lstmDropoutPercent, batch_first=True)

        #attention
        self.attn = nn.Linear(2 * config.lstmSize, config.lstmSize)
        self.attn2 = nn.Linear(config.lstmSize, config.lstmSize)

        #simple feed forward for final part
        self.fc1 = nn.Linear(config.lstmSize, config.width)
        #self.fc1 = nn.Linear(config.convSizes[-1], config.width)
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
        #embedded = F.normalize(embedded, p=2, dim=2)
        embedded = self.dropout(embedded)
        #replace the hash with the embedded vector
        embedded = torch.cat((embedded, infoset[:,:, 1:].float()), 2)
        del infoset

        """
        x = torch.transpose(embedded, 1, 2)
        for i, convs in enumerate(self.convLayers):
            for j, conv in enumerate(convs):
                x = conv(x)
                x = self.convBatchNorms[i][j](x)
                x = F.relu(x)
            kernelSize = min(max(1, x.shape[2] // config.poolSizes[i]), x.shape[2])
            x = F.max_pool1d(x, kernelSize)

        x = torch.transpose(x, 1, 2)
        #x = x.squeeze(2)

        #with convolutions, the lengths of the sequences are going to be messed up anyway
        #so it's probably fine to just take all sequences in a batch to be the same length
        """

        #"""
        #lengths are passed in if we have to worry about padding
        #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        if lengths is not None:
            lstmInput = torch.nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True)
        else:
            lstmInput = embedded
        #"""

        #remember that we set batch_first to be true
        #"""
        x, _ = self.lstm(lstmInput)
        #"""
        """
        x, _ = self.lstm(x)
        """

        #get the final output of the lstm
        #"""
        if lengths is not None:
            #have to account for padding/packing
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            lasts = x[torch.arange(0, x.shape[0]), lengths-1]
        else:
            lasts = x[:,-1]
        #"""
        """
        lasts = x[:,-1]
        """

        if trace:
            print('lasts', lasts, file=sys.stderr)
        #use both input and output of lstm to get attention weights
        #keep the batch size, but add extra dimension for sequence length
        lasts = lasts[:, None, :]
        #repeat the last output so it matches the sequence length
        lasts = lasts.repeat(1, x.shape[1], 1)
        #feed the output of the lstm appended with the final output to the attention layer
        xWithContext = torch.cat([x, lasts], 2)
        outattn = F.relu(self.attn(xWithContext))
        outattn = self.attn2(outattn)
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
        #x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x) + x)
        #x = F.relu(self.fc3(x))
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

Net = LstmNet
#Net = SimpleNet

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
        #self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.patience = 10
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)

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
        #device = torch.device('cpu')
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
        #self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)
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

        lowestLoss = 999
        lowestLossIndex =  -1

        print(file=sys.stderr)
        for j in range(epochs):
            if epochs > 1:
                print('\repoch', j, end=' ', file=sys.stderr)
            dataset = dataStorage.Dataset(self.name, self.sharedDict, self.outputSize)

            #validation split based on 
            #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
            valSplit = 0.1
            indices = list(range(dataset.size))
            split = int(np.floor(valSplit * min(dataset.size, config.epochMaxNumSamples)))
            np.random.shuffle(indices)
            trainIndices, testIndices = indices[split:min(dataset.size, config.epochMaxNumSamples)], indices[:split]
            trainSampler = SubsetRandomSampler(trainIndices)
            testSampler = SubsetRandomSampler(testIndices)

            loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=False, num_workers=config.numWorkers, pin_memory=False, collate_fn=myCollate, sampler=trainSampler)
            testLoader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=False, num_workers=config.numWorkers, pin_memory=False, collate_fn=myCollate, sampler=testSampler)

            if j == 0:
                print('training size:', len(trainIndices), 'val size:', len(testIndices), file=sys.stderr)

            totalLoss = 0

            i = 1
            sampleCount = 0
            chunkSize = dataset.size  / (miniBatchSize * 10)
            for data, dataLengths, labels, iters in loader:
                sampleCount += dataLengths.shape[0]
                i += 1

                labels = labels.float().to(device)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)

                #evaluate on network
                self.optimizer.zero_grad()
                ys = self.net(data, lengths=dataLengths, trace=False)

                #loss function from the paper
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))# / labels.shape[0]
                #if i % 10 == 0:
                    #print('----------', file=sys.stderr)
                    #print('iters', iters, file=sys.stderr)
                    #print('infosets', data, file=sys.stderr)
                    #print('ys', ys, file=sys.stderr)
                    #print('labels', labels, file=sys.stderr)

                #get gradient of loss
                #use amp because nvidia said it's better
                with amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                #loss.backward()


                #clip gradient norm, which was done in the paper
                nn.utils.clip_grad_norm_(self.net.parameters(), 1)

                #train the network
                self.optimizer.step()
                totalLoss += loss.item()

            avgLoss = totalLoss / sampleCount
            with open('trainloss.csv', 'a') as file:
                print(avgLoss, end=',', file=file)

            #get validation loss
            self.net.train(False)
            totalValLoss = 0
            valCount = 0
            for data, dataLengths, labels, iters in testLoader:
                labels = labels.float().to(device)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)
                ys = self.net(data, lengths=dataLengths, trace=False)
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))# / labels.shape[0]
                totalValLoss += loss.item()
                valCount += dataLengths.shape[0]

            self.net.train(True)

            avgValLoss = totalValLoss / valCount

            #we could use training loss
            schedLoss = avgValLoss

            if config.useScheduler:
                self.scheduler.step(schedLoss)

            if schedLoss < lowestLoss:
                lowestLoss = schedLoss
                lowestLossIndex = j

            if j - lowestLossIndex > 50:#avoid saddle points
                #self.optimizer = optim.Adam(self.net.parameters(), lr=config.learnRate)
                self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)



            #show in console and output to csv
            print('val Loss', avgValLoss, end='', file=sys.stderr)
            with open('valloss.csv', 'a') as file:
                print(totalValLoss / valCount, end=',', file=file)

        with open('valloss.csv', 'a') as file:
            print(file=file)
        with open('trainloss.csv', 'a') as file:
            print(file=file)
        print('\n', file=sys.stderr)

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
