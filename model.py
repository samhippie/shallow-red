#!/usr/bin/env python3

#from apex import amp
from hashembed.embedding import HashEmbedding
import io
import math
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
import torchvision.transforms as transforms
from batchgenerators.dataloading import MultiThreadedAugmenter
import time

import dataStorage
import config

#amp_handle = amp.init()

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

def batchNorm(x):
    if x.shape[0] == 1:
        return x
    #this is normalizing to zero mean, unit variance
    mean = torch.mean(x, dim=0)
    x = x - mean.unsqueeze(0).repeat(x.shape[0], 1)
    std = torch.std(x, dim=0)
    #if a std is 0, then the mean is also 0, so there's nothing we can do except pass 0 along via 0 / 0.01
    #and this operation is much simpler than trying to only change the 0 values
    std = std + torch.Tensor([0.00001]).repeat(std.shape[0]).cuda()
    x = x / std.unsqueeze(0).repeat(x.shape[0], 1)
    return x

class SimpleNet(nn.Module):
    def __init__(self, softmax=False):
        super(Net, self).__init__()
        self.outputSize = config.game.numActions
        self.softmax = softmax
        self.inputSize = 15
        dropout = 0
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

        self.fcVal1 = nn.Linear(config.width, config.width)
        self.fcVal2 = nn.Linear(config.width, config.width)
        self.fcValOut = nn.Linear(config.width, 1)


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

        #embedding seems to spit out some pretty low-magnitude vectors
        #so let's try normalizing
        embedded = F.normalize(embedded, p=2, dim=2)

        if trace:
            print('embedded', embedded, file=sys.stderr)

        #embedded = self.dropE(embedded)
        #replace the hash with the embedded vector
        embedded = torch.cat((embedded, infoset[:,:, 1:].float()), 2)
        del infoset

        x = F.pad(embedded, (0,0, 0,self.inputSize - embedded.shape[1]))
        x = torch.cat(tuple(x.transpose(0,1)), 1)
        x = self.dropE(x)

        if trace:
            print('before first linear', x, file=sys.stderr)

        x = F.relu(self.fc1(x))
        x = self.drop1(x)

        if trace:
            print('before split', x, file=sys.stderr)

        xVal = x

        #2 and 3 have skip connections, as they have the same sized input and output
        x = F.relu(self.fc2(x))# + x)
        #x = F.relu(self.fc2(x))
        x = self.drop2(x)

        x = F.relu(self.fc3(x))# + x)
        #x = F.relu(self.fc3(x))
        x = self.drop3(x)

        if trace:
            print('pre norm', x, file=sys.stderr)

        #deep cfr does normalization here
        #x = batchNorm(x)

        if trace:
            print('post norm', x, file=sys.stderr)

        x = self.fc6(x)

        if self.softmax:
            x = F.softmax(x, dim=1)
        else:
            pass
            #x = torch.sigmoid(x)
            #x = F.relu(x)

        #value output
        xVal = F.relu(self.fcVal1(xVal))# + xVal)
        xVal = F.relu(self.fcVal2(xVal))# + xVal)
        #xVal = torch.tanh(self.fcValOut(xVal))
        #xVal = batchNorm(xVal)
        xVal = self.fcValOut(xVal)

        if isSingle:
            return torch.cat([x[0], xVal[0]])
        else:
            x = torch.cat([x, xVal], dim=1)
            #x = batchNorm(x)
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

        self.fcVal1 = nn.Linear(config.lstmSize, config.width)
        self.fcVal2 = nn.Linear(config.width, config.width)
        self.fcValOut = nn.Linear(config.width, 1)

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
        #embedding seems to spit out some pretty low-magnitude vectors
        #so let's try normalizing
        #embedded = F.normalize(embedded, p=2, dim=2)
        if trace:
            print('embedded', embedded, file=sys.stderr)

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

        xVal = x
        x = F.relu(self.fc1(x))
        #2 and 3 have skip connections, as they have the same sized input and output
        #x = F.relu(self.fc2(x) + x)
        x = F.relu(self.fc2(x))
        #x = F.relu(self.fc3(x) + x)
        x = F.relu(self.fc3(x))
        #deep cfr does normalization here
        #I'm not sure if this is the right kind of normalization
        #x = F.normalize(x, p=2, dim=1)
        x = self.fc6(x)

        if self.softmax:
            x = F.softmax(x, dim=1)

        #value output
        xVal = F.relu(self.fcVal1(xVal))
        xVal = F.relu(self.fcVal2(xVal))
        xVal = self.fcValOut(xVal)

        if isSingle:
            return torch.cat([x[0], xVal[0]])
        else:
            x = torch.cat([x, xVal], dim=1)
            return x




Net = LstmNet
#Net = SimpleNet


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


class DeepCfrModel:

    #for advantages, the input is the state vector
    #and the output is a vector of each move's advantage
    #for strategies, the input is the state vector
    #and the output is a vector of each move's probability

    #so the inputs are exactly the same (modelInput.stateSize), and the outputs
    #are almost the same (modelInput.numActions)
    #strategy is softmaxed, advantage is not

    def __init__(self, name, softmax, writeLock, sharedDict, useNet=True):
        self.softmax = softmax
        self.lr = config.learnRate
        self.writeLock = writeLock
        self.sharedDict = sharedDict
        self.outputSize = config.game.numActions

        if(useNet):
            self.net = Net(softmax=softmax)
            self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)

        #cache of (infoset tensor, label tensor, iteration) tuples
        #will eventually be put in training db
        self.sampleCacheSize = config.sampleCacheSize
        self.sampleCache = []

        self.name = name

    def shareMemory(self):
        self.net.share_memory()

    def addSample(self, infoset, label, iter, expValue):
        #infosetTensor = np.array([hash(token) for token in infoset], dtype=np.long)
        infosetTensor = infosetToTensor(infoset)

        labelTensor = np.zeros(self.outputSize + 1)
        for action, value in label:
            #print(action, value)
            n = config.game.enumAction(action)
            labelTensor[n] = value
        labelTensor[-1] = expValue
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

    #infosets is a list of tensors
    def batchPredict(self, infosets, convertToTensor=True, trace=False):
        #print('infosets', infosets)
        batch = [infosetToTensor[i] for i in infosets] if convertToTensor else infosets
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.net = self.net.to(device)
        batch = [b.to(device) for b in batch]
        if len(batch) > 1:
            #sort, but need to keep trach of the original indices
            batch = list(enumerate(batch))
            batch.sort(key=lambda x: len(x[1]), reverse=True)
            indices = torch.tensor([b[0] for b in batch], dtype=torch.long).to(device)
            #print('indices', indices)
            #no longer need indices in batch
            batch = [b[1] for b in batch]
            #print('batch after sorting', batch, file=sys.stderr)
            #sort by length and padd
            lengths = [len(b) for b in batch]
            padded = torch.zeros(len(batch), max(lengths), len(batch[0][0]), dtype=torch.long).to(device)
            for i, d in enumerate(batch):
                end = lengths[i]
                padded[i, :end] = d[:end]
            #pass padded to network
            lengths = torch.tensor(lengths).to(device)
            #padded = padded.to(device)
            #lengths = lengths.to(device)
            #print('padded', padded, file=sys.stderr)
            #print('lengths', lengths, file=sys.stderr)
            out = self.net(padded, lengths=lengths, trace=trace)
            #unsort with scatter
            unsortedOut = torch.zeros(out.shape).to(device)
            #print('out', out)
            indices = indices.unsqueeze(1).expand(-1, out.shape[1])
            #print('expaned indices', indices)
            unsortedOut.scatter_(0, indices, out)
            return unsortedOut.cpu()
        else:
            batch[0] = batch[0].to(device)
            out = self.net(batch[0], trace=trace)
            return out.unsqueeze(0)


    def predict(self, infoset, convertToTensor=True, trace=False):
        data = infosetToTensor(infoset) if convertToTensor else infoset
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.net = self.net.to(device)
        data = data.to(device)
        data = self.net(data, trace=trace).cpu().detach().numpy()
        return data[0:-1], data[-1]

    def train(self, epochs=1):
        #move from write cache to db
        self.clearSampleCache()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        if config.newIterNets:
            newNet = Net(softmax=self.softmax)
            #embedding should be preserved across iterations
            #but we want a fresh start for the strategy
            #newNet.embeddings = self.net.embeddings
            #maybe not, if we're going to be using the old net in the future
            self.net = newNet


        self.net = self.net.to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)
        miniBatchSize = config.miniBatchSize
        self.net.train(True)

        #used for scheduling
        lowestLoss = 999
        lowestLossIndex =  -1
        lastResetLoss = None
        runningLoss = []

        #we don't really use the dataset, but we use it to read some files
        #we should fix this, but it works and doesn't really hurt anything
        dataset = dataStorage.Dataset(self.name, self.sharedDict, self.outputSize)

        #validation split based on 
        #https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
        indices = list(range(dataset.size))
        split = int(np.floor(config.valSplit * min(dataset.size, config.epochMaxNumSamples)))
        np.random.shuffle(indices)
        trainIndices, testIndices = indices[split:min(dataset.size, config.epochMaxNumSamples)], indices[:split]
        #trainSampler = SubsetRandomSampler(trainIndices)
        #testSampler = SubsetRandomSampler(testIndices)

        trainingLoader = dataStorage.BatchDataLoader(id=self.name, indices=trainIndices, batch_size=config.miniBatchSize, num_threads_in_mt=config.numWorkers)
        if config.numWorkers > 1:
            trainingLoader = MultiThreadedAugmenter(trainingLoader, None, config.numWorkers, 2, None)

        testingLoader = dataStorage.BatchDataLoader(id=self.name, indices=trainIndices, batch_size=config.miniBatchSize, num_threads_in_mt=config.numWorkers)
        if config.numWorkers > 1:
            testingLoader = MultiThreadedAugmenter(testingLoader, None, config.numWorkers)

        print(file=sys.stderr)
        for j in range(epochs):
            if epochs > 1:
                print('\repoch', j, end=' ', file=sys.stderr)

            """
            if j % 100 == 0:
                self.net.train(False)
                exampleInfoSets = [
                    ['start', 'hand', '2', '0', 'deal', '1', 'raise'],
                    ['start', 'hand', '9', '0', 'deal', '1', 'raise'],
                    ['start', 'hand', '14', '0', 'deal', '1', 'raise'],
                ]
                self.net.train(True)

                for example in exampleInfoSets:
                    print('example input:', example, file=sys.stderr)
                    probs, expVal = self.predict(example, trace=False)
                    print('exampleOutput (deal, fold, call, raise)', np.round(100 * probs), 'exp value', round(expVal * 100), file=sys.stderr)
            """

            
            #loader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, shuffle=False, num_workers=config.numWorkers, pin_memory=False, collate_fn=myCollate, sampler=trainSampler)

            if j == 0:
                print('training size:', len(trainIndices), 'val size:', len(testIndices), file=sys.stderr)

            totalLoss = 0
            lossFunc = nn.MSELoss()

            i = 1
            sampleCount = 0
            chunkSize = dataset.size  / (miniBatchSize * 10)
            #for data, dataLengths, labels, iters in loader:
            for data, dataLengths, labels, iters in trainingLoader:
                sampleCount += dataLengths.shape[0]
                i += 1

                labels = labels.float().to(device)
                #labels = batchNorm(labels)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)
                #print('------')
                #print('data', data.squeeze(), file=sys.stderr)
                #print('label', labels, file=sys.stderr)
                #print('label exp values', labels[:, -1], file=sys.stderr)
                #print('label exp values avg', labels[:, -1].mean(), file=sys.stderr)
                #print('label exp values std', labels[:, -1].std(), file=sys.stderr)

                #evaluate on network
                self.optimizer.zero_grad()
                ys = self.net(data, lengths=dataLengths, trace=False)
                #print('ys', ys, file=sys.stderr)
                #print('ys exp values', ys[:, -1], file=sys.stderr)
                #print('ys exp values avg', ys[:, -1].mean(), file=sys.stderr)
                #print('ys exp values std', ys[:, -1].std(), file=sys.stderr)

                #loss function from the paper
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))# / labels.shape[0]
                #loss = torch.sum((labels - ys) ** 2) / labels.shape[0]
                #loss = lossFunc(ys, labels)
                #print('loss', loss)
                """
                if i % 1 == 0:
                    print('----------', file=sys.stderr)
                    #print('iters', iters, file=sys.stderr)
                    print('infosets', data, file=sys.stderr)
                    print('ys', torch.round(ys * 100) / 100, file=sys.stderr)
                    print('labels', torch.round(labels * 100) / 100, file=sys.stderr)
                """

                #get gradient of loss
                #use amp because nvidia said it's better
                #TODO fix amp
                #with amp_handle.scale_loss(loss, self.optimizer) as scaled_loss:
                    #scaled_loss.backward()
                loss.backward()


                #clip gradient norm, which was done in the paper
                nn.utils.clip_grad_norm_(self.net.parameters(), 5)

                #train the network
                self.optimizer.step()
                totalLoss += loss.item()


            avgLoss = totalLoss / sampleCount
            with open('trainloss.csv', 'a') as file:
                print(avgLoss, end=',', file=file)

            #get validation loss
            #testLoader = torch.utils.data.DataLoader(dataset, batch_size=miniBatchSize, num_workers=config.numWorkers, collate_fn=myCollate, sampler=testSampler)
            self.net.train(False)
            totalValLoss = 0
            valCount = 0
            #for data, dataLengths, labels, iters in testLoader:
            for data, dataLengths, labels, iters in testingLoader:
                labels = labels.float().to(device)
                #print('labels', np.round(100 * labels.cpu().numpy()) / 100, file=sys.stderr)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)
                ys = self.net(data, lengths=dataLengths, trace=False)
                #print('ys', np.round(100 * ys.cpu().detach().numpy()) / 100, file=sys.stderr)
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2))# / labels.shape[0]
                totalValLoss += loss.item()
                valCount += dataLengths.shape[0]

            self.net.train(True)

            avgValLoss = totalValLoss / valCount

            #running average of last 3 validation losses
            runningLoss.append(avgValLoss)
            if len(runningLoss) > 3:
                runningLoss = runningLoss[-3:]
            schedLoss = sum(runningLoss) / len(runningLoss)

            if config.useScheduler:
                self.scheduler.step(schedLoss)

            if schedLoss < lowestLoss:
                lowestLoss = schedLoss
                lowestLossIndex = j

            if j - lowestLossIndex > 3 * config.schedulerPatience:#avoid saddle points
                print('resetting learn rate to default', j, lowestLossIndex, lowestLoss, schedLoss, lastResetLoss, file=sys.stderr)
                self.optimizer = optim.Adam(self.net.parameters(), lr=config.learnRate)
                #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
                self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)
                lowestLossIndex = j

                #if we've reset before and made no progress, just stop
                if lastResetLoss is not None and (schedLoss - lastResetLoss) / lastResetLoss > -0.01:
                    print('stopping epoch early, (schedLoss - lastResetLoss) / lastResetLoss) is', (schedLoss - lastResetLoss) / lastResetLoss, file=sys.stderr)
                    break
                lastResetLoss = schedLoss



            #show in console and output to csv
            print('val Loss', avgValLoss, end='', file=sys.stderr)
            with open('valloss.csv', 'a') as file:
                #print(avgValLoss, end=',', file=file)
                print(schedLoss, end=',', file=file)

        with open('valloss.csv', 'a') as file:
            print(file=file)
        with open('trainloss.csv', 'a') as file:
            print(file=file)
        print('\n', file=sys.stderr)

        self.net.train(False)
        #warPoker examples
        """
        exampleInfoSets = [
            ['start', 'hand', '2', '0', 'deal', '1', 'raise'],
            ['start', 'hand', '7', '0', 'deal', '1', 'raise'],
            ['start', 'hand', '14', '0', 'deal', '1', 'raise'],
            ['start', 'hand', '2', '1', 'deal'],
            ['start', 'hand', '7', '1', 'deal'],
            ['start', 'hand', '14', '1', 'deal'],
        ]
        for example in exampleInfoSets:
            print('example input:', example, file=sys.stderr)
            probs, expVal = self.predict(example, trace=False)
            print('exampleOutput (deal, fold, call, raise)', np.round(100 * probs), 'exp value', round(expVal * 100), file=sys.stderr)
        """

        #ace example
        """
        target = infosetToTensor(exampleInfoSets[2]).squeeze()
        count = 0
        total = None
        for i in range(len(dataset)):
            infoset, label, iter = dataset[i]
            infoset = infoset.squeeze()
            if infoset.shape[0] == target.shape[0] and torch.all(torch.eq(infoset, target)):
                print(label, file=sys.stderr)
                if count == 0:
                    total = label
                else:
                    total = total + label
                count += 1
        if count > 0:
            print('average', total / count)
            input("PRESS ENTER (this will work or crash, either is fine)")
        """

        #clean old data out
        #dataStorage.clearSamplesByName(self.name)
