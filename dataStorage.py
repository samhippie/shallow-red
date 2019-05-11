#!/usr/bin/env python3

import asyncio
import numpy as np
import os
import os.path
import pickle
import sys
import torch
import torch.utils.data
import torch.multiprocessing as mp
from batchgenerators.dataloading import SlimDataLoaderBase
from lockfile import LockFile

import config

#this manages the training data for deep cfr

#directory where samples are stored
DATA_DIR = config.dataDir
#DATA_DIR = './data/'

#whether to store data in memory or on disk
IN_MEMORY = config.inMemory

#whether to cache each sample in-memory on read
#this only has an effect when IN_MEMORY is False
BIG_CACHE = config.bigCache

#deletes the data from DATA_DIR
#does not delete the folder itself
def clearData():
    os.system('rm -r ' + DATA_DIR + '*')
    os.system('rm valloss.csv')
    os.system('rm trainloss.csv')

#deletes data belonging to a certain name
def clearSamplesByName(name):
    #remove the entry in index
    if os.path.exists(DATA_DIR + 'index'):
        target = -1
        #find the line
        with open(DATA_DIR + 'index', 'r') as file:
            lines = list(file.readlines())
            for i in range(len(lines)):
                if lines[i].split(',')[0] == name:
                    target = i
                    break
        #write out all lines except the targeted one
        if target != -1:
            del lines[i]
            with open(DATA_DIR + 'index', 'w') as file:
                for line in lines:
                    print(line, file=file, end='')

    #delete the data files
    os.system('rm -r ' + DATA_DIR + name + '/*')

#lock is a multiprocess manager lock
#id determines which dataset the samples belong to
#samples is a list of numpy arrays
#the nth sample will be written to data/id/n
#shared dict is used for any shared data (which will probably only include in-memory datasets)
def addSamples(lock, id, samples, sharedDict):
    #write our count to the index file
    #which must be thread-safe
    #actually lock on the index file
    lock = LockFile(DATA_DIR + 'index')
    lock.acquire()
    if IN_MEMORY:
        if 'smp' + id not in sharedDict:
            sharedDict['smp' + id] = samples
        else:
            old = sharedDict['smp' + id]
            sharedDict['smp' + id] = old + samples
    else:
        lines = []
        count = 0
        if not os.path.exists(DATA_DIR):
            os.mkdir(DATA_DIR)
        #get the current number of samples
        if os.path.exists(DATA_DIR + 'index'):
            with open(DATA_DIR + 'index', 'r') as file:
                lines = list(file.readlines())
                for i in range(len(lines)):
                    line = lines[i]
                    if line.split(',')[0] == id:
                        count = int(line.split(',')[1][:-1])
                        lines[i] = id + ',' + str(count + len(samples)) + '\n'
                        break

        #brand new sample set
        if count == 0:
            lines.append(id + ',' + str(len(samples)) + '\n')
        if config.maxNumSamples[id] and count >= config.maxNumSamples[id]:
            replace = True
            #we're replacing existing files, so no need to update the index
        else:
            replace = False
            #update the indices after we're written our files
            with open(DATA_DIR + 'index', 'w') as file:
                for line in lines:
                    print(line, file=file, end='')

    #replacement means no amount of concurrency is safe here

    #we really should support replacement of in-memory samples
    if not IN_MEMORY:
        if replace:
            #pick which samples get removed
            #we need to give old and new samples an equal chance so we don't introduce bias
            writeIndices = np.random.choice(count + len(samples), len(samples), replace=False)
            for i, sample in zip(writeIndices, samples):
                #if we're supposed to replace a sample, then don't save that sample
                if i < count:
                    with open(DATA_DIR + id + '/' + str(i), 'wb+') as file:
                        pickle.dump(sample, file)
        else:
            #write our each sample to its own file
            if not os.path.exists(DATA_DIR + id):
                os.mkdir(DATA_DIR + id)
            for i in range(len(samples)):
                with open(DATA_DIR + id + '/' + str(count + i), 'wb+') as file:
                    pickle.dump(samples[i], file)

    lock.release()


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



#following this
#https://github.com/MIC-DKFZ/batchgenerators/blob/master/batchgenerators/examples/multithreaded_dataloading.py
class BatchDataLoader(SlimDataLoaderBase):
    def __init__(self, id, indices, batch_size, num_threads_in_mt):
        super(BatchDataLoader, self).__init__(None, batch_size, num_threads_in_mt)

        self.id = id
        self.size = len(indices)
        self.indices = indices
        self.current_position = 0
        self.was_initialized = False

    def reset(self):
        self.current_position = self.thread_id
        self.was_initialized = True

    def generate_train_batch(self):
        if not self.was_initialized:
            self.reset()
        batch = []
        for i in range(self.batch_size):
            if self.current_position < self.size:
                idx = self.indices[self.current_position]
                self.current_position += self.number_of_threads_in_multithreaded
                with open(DATA_DIR + self.id + '/' + str(idx), 'rb') as file:
                    sample = pickle.load(file)

                data, label, iter = sample
                label = torch.from_numpy(label)
                iter = torch.from_numpy(iter)
                batch.append([data, label, iter])

            elif len(batch) > 0:
                return myCollate(batch)
            else:
                self.reset()
                raise StopIteration
        return myCollate(batch)
            

class Dataset(torch.utils.data.Dataset):
    def __init__(self, id, sharedDict, outputSize):
        #print('initing dataloader', file=sys.stderr)
        self.id = id
        if IN_MEMORY:
            #print('initing in memory', file=sys.stderr)
            self.sharedDict = sharedDict
            if 'smp' + id not in sharedDict:
                self.size = 0
                self.samples = []
            else:
                self.samples = sharedDict['smp' + id]
                self.size = self.samples.shape[0]
        else:
            #print('initing on disk', file=sys.stderr)
            #self.sampleCache = {}
            with open(DATA_DIR + 'index', 'r') as file:
                for line in file.readlines():
                    if line.split(',')[0] == id:
                        self.size = int(line.split(',')[1][:-1])

    def __getitem__(self, idx):
        if IN_MEMORY:
            #print('getting sample from memory', file=sys.stderr)
            sample = self.samples[idx]
            #print('got sample from memory', file=sys.stderr)
        else:
            #print('getting sample from disk', file=sys.stderr)
            #if idx not in self.sampleCache:
            with open(DATA_DIR + self.id + '/' + str(idx), 'rb') as file:
                #self.sampleCache[idx] = np.load(file)
                #sample = np.load(file)
                sample = pickle.load(file)
            #sample = self.sampleCache[idx]
            #print('got sample from disk', file=sys.stderr)

        #data = sample[0:modelInput.stateSize]
        #label = sample[modelInput.stateSize:modelInput.stateSize + modelInput.numActions]

        data, label, iter = sample

        #data = sample[0:-(self.outputSize + 1)]
        #data is a python list
        #(except when it isn't)
        #data = torch.from_numpy(data)

        #label = sample[-(self.outputSize + 1):-1]
        label = torch.from_numpy(label)

        #iter = sample[-1:]
        iter = torch.from_numpy(iter)

        return data, label, iter

    def __len__(self):
        return self.size

def runner(lock, rank):
    async def task(rank):
        addSamples(lock, 'test', [np.array([rank] * 1000)])

    loop = asyncio.get_event_loop()
    loop.run_until_complete(task(rank))

def main():
    print('starting')

    m = mp.Manager()
    lock = m.Lock()
    processes = []
    for rank in range(4):
        p = mp.Process(target=runner, args=(lock, rank,))
        p.start()
        processes.append(p)
        print('started')

    print('waiting for processes to finish')
    for p in processes:
        p.join()
        print('join')
    print('done')

if __name__ == '__main__':
    main()

