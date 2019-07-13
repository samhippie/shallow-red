#!/usr/bin/env python3

from apex import amp
from hashembed.embedding import HashEmbedding
import io
import itertools
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

AMP_OPT_LEVEL = "O2"

if config.optimizer == 'adam':
    OPTIMIZER = optim.Adam
elif config.optimizer == 'sgd':
    OPTIMIZER = optim.SGD

#how many bits are used to represent numbers in tokens
NUM_TOKEN_BITS = config.numTokenBits

#not actually binary, but some weird logarithmic unary-ish thing
def numToBinary(n):
    b = []
    for i in range(NUM_TOKEN_BITS):
        if n <= 0:
            b.append(0)
        elif n >= 1 << i:
            b.append(1)
        else:
            #b.append(n / (1 << i))#this is neat, but I'd have to convert a bunch of longs to floats
            b.append(0)
            n = 0
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

        if config.enableCnn:
            #using these numbers reduces the total input size going to the lstm to around 1/3
            #while reducing the length of the sequence to around 1/8
            #(this was writtine with 2,3,4 conv sizes)
            convSize = config.embedSize + config.numTokenBits
            lstmInputSize = 4 * convSize
            self.conv1 = nn.Conv1d(convSize, 2 * convSize, kernel_size=11, stride=2, padding=6)
            self.conv1Dropout = nn.Dropout(0.2)
            #self.bn1 = nn.BatchNorm1d(convSize)
            self.conv2 = nn.Conv1d(2 * convSize, 3 * convSize, kernel_size=11, stride=2, padding=6)
            self.conv2Dropout = nn.Dropout(0.2)
            #self.bn2 = nn.BatchNorm1d(convSize)
            self.conv3 = nn.Conv1d(3 * convSize, 4 * convSize, kernel_size=11, stride=2, padding=6)
            self.conv3Dropout = nn.Dropout(0.2)
        else:
            lstmInputSize = config.embedSize + config.numTokenBits

        #'LSTM' to process infoset via the embeddings
        self.lstm = nn.LSTM(lstmInputSize, config.lstmSize, num_layers=config.numLstmLayers, dropout=config.lstmDropoutPercent, bidirectional=False, batch_first=True)

        #attention
        if config.enableAttention:
            self.attn1 = nn.Linear(2 * config.lstmSize, config.lstmSize)
            #self.attn2 = nn.Linear(2 * config.lstmSize, config.lstmSize)
            #self.attn3 = nn.Linear(2 * config.lstmSize, config.lstmSize)

        self.lstmDropout = nn.Dropout(0)

        #simple feed forward for final part
        self.fc1 = nn.Linear(config.lstmSize, config.width)
        #if we want to skip the lstm, I think
        #which we don't currently support
        #self.fc1 = nn.Linear(config.convSizes[-1], config.width)
        self.fc2 = nn.Linear(config.width, config.width)
        self.fc3 = nn.Linear(config.width, config.width)
        self.fc6 = nn.Linear(config.width, self.outputSize)

        self.fcVal1 = nn.Linear(config.lstmSize, config.width)
        self.fcVal2 = nn.Linear(config.width, config.width)
        self.fcValOut = nn.Linear(config.width, 1)

    def forward(self, infoset, lengths=None, trace=False):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

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
        x = torch.cat((embedded, infoset[:,:, 1:].to(dtype=embedded.dtype)), 2)
        #del infoset

        #go through a couple conv layers
        #need to mask out the part proportional to the initial lengths in the conv output
        #so we can keep the garbage out here and in the lstm layer
        #if lengths is None, then there is no padding and no point in masking
        if config.enableCnn:
            x = torch.transpose(x, 1, 2)
            preLength = x.shape[2]
            x = F.relu(self.conv1(x))
            #x = self.bn1(x)
            x = self.conv1Dropout(x)

            if lengths is not None:
                lengths = lengths.float()
                lengths *= x.shape[2] / preLength
                #give output a little extra room
                lengths += 1
                #need lengths to still be valid lengths
                lengths = torch.clamp(lengths, min=1, max=x.shape[2])
                lengths = lengths.long()
                x = torch.transpose(x, 1, 2)
                mask = torch.arange(x.shape[1], device=device)[None, :] >= lengths[:, None]
                x[mask] = 0
                x = torch.transpose(x, 1, 2)
                preLength = x.shape[2]

            x = F.relu(self.conv2(x))
            #x = self.bn2(x)
            x = self.conv2Dropout(x)

            if lengths is not None:
                lengths = lengths.float()
                lengths *= x.shape[2] / preLength
                lengths += 1
                lengths = torch.clamp(lengths, min=1, max=x.shape[2])
                lengths = lengths.long()
                x = torch.transpose(x, 1, 2)
                mask = torch.arange(x.shape[1], device=device)[None, :] >= lengths[:, None]
                x[mask] = 0
                del mask
                x = torch.transpose(x, 1, 2)

            x = F.relu(self.conv3(x))
            x = self.conv2Dropout(x)

            if lengths is not None:
                lengths = lengths.float()
                lengths *= x.shape[2] / preLength
                lengths += 1
                lengths = torch.clamp(lengths, min=1, max=x.shape[2])
                lengths = lengths.long()
                x = torch.transpose(x, 1, 2)
                mask = torch.arange(x.shape[1], device=device)[None, :] >= lengths[:, None]
                x[mask] = 0
                del mask
            else:
                x = torch.transpose(x, 1, 2)

        #lengths are passed in if we have to worry about padding
        #https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)

        #remember that we set batch_first to be true
        x, _ = self.lstm(x)

        #get the final output of the lstm

        if lengths is not None:
            #have to account for padding/packing
            #don't use the lengths this returns, as it put it on the cpu instead of cuda
            #and we need the lengths for cuda stuff
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
            lasts = x[torch.arange(0, x.shape[0], device=device), lengths-1]
        else:
            lasts = x[:,-1]

        if trace:
            print('lasts', lasts, file=sys.stderr)

        if config.enableAttention:
            #use both input and output of lstm to get attention weights
            #keep the batch size, but add extra dimension for sequence length
            lasts = lasts[:, None, :]
            #repeat the last output so it matches the sequence length
            lasts = lasts.repeat(1, x.shape[1], 1)
            #feed the output of the lstm appended with the final output to the attention layer
            xWithContext = torch.cat([x, lasts], 2)
            outattn1 = self.attn1(xWithContext)
            #outattn2 = self.attn2(xWithContext)
            #outattn3 = self.attn3(xWithContext)
            #mask out the padded values (cnns don't have padded values)
            if lengths is not None:
                #http://juditacs.github.io/2018/12/27/masked-attention.html
                #we're setting padding values to -inf to get softmaxed to 0, so we want to mask out the real data
                #hence >= instead of >
                mask = torch.arange(outattn1.shape[1], device=device)[None, :] >= lengths[:, None]
                outattn1[mask] = float('-inf')
                #outattn2[mask] = float('-inf')
                #outattn3[mask] = float('-inf')

            #softmax so the weights for each element of each output add up to 1
            outattn1 = F.softmax(outattn1, dim=1)
            #outattn2 = F.softmax(outattn2, dim=1)
            #outattn1 = F.softmax(outattn3, dim=1)
            #sum along the sequence
            x = torch.sum(x * outattn1, dim=1)
            #x2 = torch.sum(x * outattn2, dim=1)
            #x3 = torch.sum(x * outattn3, dim=1)
            #x = torch.cat([x1, x2, x3], dim=1)
        else:
            x = lasts

        x = self.lstmDropout(x)

        if trace:
            print('x to fc', x, file=sys.stderr)

        xVal = x
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

        #value output
        xVal = F.relu(self.fcVal1(xVal))
        xVal = F.relu(self.fcVal2(xVal) + xVal)
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

    def __init__(self, name, softmax, writeLock, sharedDict, saveFile=None, useNet=True):
        self.softmax = softmax
        self.lr = config.learnRate
        self.writeLock = writeLock
        self.sharedDict = sharedDict
        self.outputSize = config.game.numActions
        self.saveFile = saveFile

        if(useNet):
            self.net = Net(softmax=softmax).cuda()
            self.optimizer = OPTIMIZER(self.net.parameters(), lr=self.lr)
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=AMP_OPT_LEVEL)
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

        #this could be cleaned up now that we're using action indices
        #fill the tensor with -1, which means that actions without labels are considered bad instead of neutral
        #this doesn't make sense for softmaxed values, but we haven't softmaxed in months
        labelTensor = -1 * np.ones(self.outputSize + 1)
        for n, value in enumerate(label):
            #print(action, value)
            #n = config.game.enumAction(action)
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

    #saving/loading models for inference
    #once we save or load a model, we shouldn't train on it
    #so we just save enough information for inference

    def saveModel(self, n):
        if self.saveFile:
            path = self.saveFile + 'model.' + self.name  + '.' + str(n) + '.pt'
            print('saving model to', path, file=sys.stderr)
            torch.save(self.net.state_dict(), path)

    def loadModel(self, n):
        if self.saveFile:
            path = self.saveFile + 'model.' + self.name  + '.' + str(n) + '.pt'
            print('loading model', path, file=sys.stderr)
            self.net.load_state_dict(torch.load(path))
            self.net.eval()

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
            out = self.net(padded, lengths=lengths, trace=trace)#.float()
            #unsort with scatter
            unsortedOut = torch.zeros(out.shape).to(device).to(dtype=out.dtype)
            #print('out', out)
            indices = indices.unsqueeze(1).expand(-1, out.shape[1])
            #print('expaned indices', indices)
            unsortedOut.scatter_(0, indices, out)
            return unsortedOut.cpu()
        else:
            batch[0] = batch[0].to(device)
            out = self.net(batch[0], trace=trace)
            return out.unsqueeze(0).float()


    def predict(self, infoset, convertToTensor=True, trace=False):
        data = infosetToTensor(infoset) if convertToTensor else infoset
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        self.net = self.net.to(device)
        data = data.to(device)
        data = self.net(data, trace=trace).float().cpu().detach().numpy()
        return data[0:-1], data[-1]

    def train(self, iteration, epochs=1):
        #move from write cache to db
        self.clearSampleCache()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')

        if config.newIterNets:
            newNet = Net(softmax=self.softmax)
            #embedding should be preserved across iterations
            #but we want a fresh start for the strategy
            #actually, the deep cfr paper said don't do this
            #newNet.load_state_dict(self.net.state_dict())
            #I'm still going to copy over the embedding
            #newNet.embeddings.load_state_dict(self.net.embeddings.state_dict())
            self.net = newNet


        self.net = self.net.to(device)
        #self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        self.optimizer = OPTIMIZER(self.net.parameters(), lr=self.lr)
        self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level=AMP_OPT_LEVEL)
        #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)
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

        #we could scale the minibatch size by the number of samples, but this slows things down
        #miniBatchSize = min(config.miniBatchSize, len(trainIndices) // config.numWorkers)
        miniBatchSize = config.miniBatchSize

        #we could instead scale the number of workers by the number of minibatches
        #numWorkers = min(config.numWorkers, len(trainIndices) // miniBatchSize)
        numWorkers = config.numWorkers

        trainingLoader = dataStorage.BatchDataLoader(id=self.name, indices=trainIndices, batch_size=miniBatchSize, num_threads_in_mt=config.numWorkers)
        baseTrainingLoader = trainingLoader
        if numWorkers > 1:
            trainingLoader = MultiThreadedAugmenter(trainingLoader, None, numWorkers, 2, None)

        testingLoader = dataStorage.BatchDataLoader(id=self.name, indices=testIndices, batch_size=miniBatchSize, num_threads_in_mt=numWorkers)
        baseTestingLoader = testingLoader
        if numWorkers > 1:
            testingLoader = MultiThreadedAugmenter(testingLoader, None, numWorkers)

        print(file=sys.stderr)
        shuffleStride = 10#TODO move to config
        for j in range(epochs):
            if epochs > 1:
                print('\repoch', j, end=' ', file=sys.stderr)

            if j == 0:
                print('training size:', len(trainIndices), 'val size:', len(testIndices), file=sys.stderr)

            totalLoss = 0
            lossFunc = nn.MSELoss()

            if (j + 1) % shuffleStride == 0:
                baseTrainingLoader.shuffle()

            i = 1
            sampleCount = 0
            chunkSize = dataset.size  / (miniBatchSize * 10)
            for data, dataLengths, labels, iters in trainingLoader:
                sampleCount += 1#dataLengths.shape[0]
                i += 1

                labels = labels.float().to(device)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)

                #evaluate on network
                self.optimizer.zero_grad()
                ys = self.net(data, lengths=dataLengths, trace=False).squeeze()

                #loss function from the paper
                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2)) / (torch.sum(iters).item())
                #get gradient of loss
                #use amp because nvidia said it's better
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
                #loss.backward()


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
            baseTestingLoader.shuffle()
            totalValLoss = 0
            valCount = 0
            stdTotal = 0
            stdCount = 0
            #for data, dataLengths, labels, iters in testLoader:
            for data, dataLengths, labels, iters in testingLoader:
                labels = labels.float().to(device)
                #print('labels', np.round(100 * labels.cpu().numpy()) / 100, file=sys.stderr)
                iters = iters.float().to(device)
                data = data.long().to(device)
                dataLengths = dataLengths.long().to(device)
                ys = self.net(data, lengths=dataLengths, trace=False).squeeze()
                if valCount == 0:
                    #print('data', data[0:min(10, len(data))])
                    print('labels', labels[0:min(10, len(labels))])
                    print('output', ys[0:min(10, len(labels))])
                    print('stddev', ys.std())
                stdTotal += ys.std().item()
                stdCount += 1

                loss = torch.sum(iters.view(labels.shape[0],-1) * ((labels - ys) ** 2)) / (torch.sum(iters).item())
                totalValLoss += loss.item()
                valCount += 1#dataLengths.shape[0]

            self.net.train(True)

            with open('stddev.csv', 'a') as file:
                print(stdTotal / stdCount, end=',', file=file)

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

            """
            if schedLoss < 0.35:
                print('eh,', schedLoss, 'is good enough', file=sys.stderr)
                break
            """

            """
            if j - lowestLossIndex > 3 * config.schedulerPatience:#avoid saddle points
                #print('resetting learn rate to default', j, lowestLossIndex, lowestLoss, schedLoss, lastResetLoss, file=sys.stderr)
                #self.optimizer = optim.Adam(self.net.parameters(), lr=config.learnRate)
                #self.optimizer = optim.SGD(self.net.parameters(), lr=self.lr, momentum=0.9)
                #self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=config.schedulerPatience, verbose=False)
                print('stopping epoch early')
                break
                lowestLossIndex = j

                #if we've reset before and made no progress, just stop
                if lastResetLoss is not None and (schedLoss - lastResetLoss) / lastResetLoss > -0.01:
                    print('stopping epoch early, (schedLoss - lastResetLoss) / lastResetLoss) is', (schedLoss - lastResetLoss) / lastResetLoss, file=sys.stderr)
                    break
                lastResetLoss = schedLoss
            """



            #show in console and output to csv
            print('val Loss', avgValLoss, end='', file=sys.stderr)
            with open('valloss.csv', 'a') as file:
                #print(avgValLoss, end=',', file=file)
                print(schedLoss, end=',', file=file)

        with open('valloss.csv', 'a') as file:
            print(file=file)
        with open('trainloss.csv', 'a') as file:
            print(file=file)
        with open('stddev.csv', 'a') as file:
            print(file=file)
        print('\n', file=sys.stderr)

        self.net.train(False)

        self.saveModel(iteration)



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

if __name__ == '__main__':
    #example of how to handle the sliced lstm model's padding
    batch = [ [[10, 11], [12,13,14,15]], [[1,2,3],[4,5],[6,7,8]] ]
    x, l0, i0, pl, l1, i1 = SlicedLstmNet.collate(batch, convertToTensor=True)
    x = torch.sum(x, dim=1)#if we went through an lstm, we'd use l0
    x = SlicedLstmNet.unsort(x, i0)
    x = SlicedLstmNet.split(x, pl)
    x = torch.sum(x, dim=1)#again, we'd use l1 here
    x = SlicedLstmNet.unsort(x, i1)
    print(x)
