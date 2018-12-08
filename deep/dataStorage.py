#!/usr/bin/env python3

import asyncio
import modelInput
import numpy as np
import os
import os.path
import torch
import torch.utils.data
import torch.multiprocessing as mp

#this manages the training data for deep cfr

#directory where samples are stored
#directory should exist, and there should be a (possibly empty) "index" file
DATA_DIR = './data/'

#lock is a multiprocess manager lock
#id determines which dataset the samples belong to
#samples is a list of numpy arrays
#the nth sample will be written to data/id/n
def addSamples(lock, id, samples):
    #write our count to the index file
    #which must be thread-safe
    lock.acquire()
    lines = []
    count = 0
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if os.path.exists(DATA_DIR + 'index'):
        with open(DATA_DIR + 'index', 'r') as file:
            lines = file.readlines()
            for i in range(len(lines)):
                line = lines[i]
                if line.startswith(id):
                    count = int(line.split(',')[1][:-1])
                    lines[i] = id + ',' + str(count + len(samples))

    #brand new sample set
    if count == 0:
        lines.append(id + ',' + str(len(samples)))
    #update the indices after we're written our files
    with open(DATA_DIR + 'index', 'w+') as file:
        for line in lines:
            print(line, file=file)

    lock.release()

    #write our each sample to its own file
    if not os.path.exists(DATA_DIR + id):
        os.mkdir(DATA_DIR + id)
    for i in range(len(samples)):
        with open(DATA_DIR + id + '/' + str(count + i), 'wb+') as file:
            np.save(file, samples[i])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, id):
        self.id = id
        with open(DATA_DIR + 'index', 'r') as file:
            for line in file.readlines():
                if line.startswith(id):
                    self.size = int(line.split(',')[1][:-1])

    def __getitem__(self, idx):
        with open(DATA_DIR + self.id + '/' + str(idx), 'rb') as file:
            sample = np.fromfile(file)
            data = sample[0:modelInput.stateSize]
            data = torch.from_numpy(data).float()
            label = sample[modelInput.stateSize:modelInput.stateSize + modelInput.numActions]
            label = torch.from_numpy(label).float()
            iter = sample[-1:]
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

