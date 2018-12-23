#!/usr/bin/env python3

import asyncio
import modelInput
import numpy as np
import os
import os.path
import pickle
import sys
import torch
import torch.utils.data
import torch.multiprocessing as mp

#this manages the training data for deep cfr

#directory where samples are stored
DATA_DIR = '/home/sam/data/'
#DATA_DIR = './data/'

#whether to store data in memory or on disk
IN_MEMORY = False

#whether to cache each sample in-memory on read
#this only has an effect when IN_MEMORY is False
BIG_CACHE = False

#deletes the data from DATA_DIR
#does not delete the folder itself
def clearData():
    os.system('rm -r ' + DATA_DIR + '*')

#lock is a multiprocess manager lock
#id determines which dataset the samples belong to
#samples is a list of numpy arrays
#the nth sample will be written to data/id/n
#shared dict is used for any shared data (which will probably only include in-memory datasets)
def addSamples(lock, id, samples, sharedDict):
    #write our count to the index file
    #which must be thread-safe
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
        #update the indices after we're written our files
        with open(DATA_DIR + 'index', 'w') as file:
            for line in lines:
                print(line, file=file, end='')

    lock.release()

    if not IN_MEMORY:
        #write our each sample to its own file
        if not os.path.exists(DATA_DIR + id):
            os.mkdir(DATA_DIR + id)
        for i in range(len(samples)):
            with open(DATA_DIR + id + '/' + str(count + i), 'wb+') as file:
                pickle.dump(samples[i], file)
                #np.save(file, samples[i])


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
        data = torch.from_numpy(data).float()

        #label = sample[-(self.outputSize + 1):-1]
        label = torch.from_numpy(label).float()

        #iter = sample[-1:]
        iter = torch.from_numpy(iter).float()

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

