#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import sys

if len(sys.argv) == 3:
    lower = int(sys.argv[1])
    upper = int(sys.argv[2])
else:
    lower = 0
    upper = None


vals = []
with open('valloss.csv') as file:
    for line in file.readlines():
        vals.append([float(x) for x in line.strip().split(',') if len(x) > 0])

trains = []
with open('trainloss.csv') as file:
    for line in file.readlines():
        trains.append([float(x) for x in line.strip().split(',') if len(x) > 0])

print('max rows:', len(vals))

if upper == None:
    upper = len(vals)

for i, (val, train) in enumerate(zip(vals[lower:upper], trains[lower:upper])):
    avgVal = []
    ep = 0.9
    cur = val[0]
    for v in val:
        cur = ep * cur + (1-ep) * v
        avgVal.append(cur)

    avgTrain = []
    cur = train[0]
    for t in train:
        cur = ep * cur + (1-ep) * t
        avgTrain.append(cur)

    plt.subplot(upper - lower, 1, i+1)
    plt.scatter(range(len(val)), val, color='blue')
    plt.scatter(range(len(train)), train, color='green')
    plt.plot(avgVal, label='val smoothed', color='orange')
    plt.plot(avgTrain, label='train smoothed', color='red')
    plt.legend(frameon=False)
    plt.title('player ' + str((i % 2) + 1) + ' epoch ' + str(i // 2))
    plt.grid()
plt.show()
