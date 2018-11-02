#!/usr/bin/env python3

import collections
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow import keras

import modelInput

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#used to compare a trained model to a basic model for the same inputs
#can also be used if we want to train a model using the behavior of a basic model
class CombinedModel:

    def __init__(self, trainedModel, basicModel):
        self.trainedModel = trainedModel
        self.basicModel = basicModel

        #t controls output of getExpValue
        #0 for basic model, 1 for trained, in between for weighted average
        self.t = 0

        self.compare = False
        self.compPointsBasic = []
        self.compPointsTrained = []

    def getExpValue(self, stateHash=None, stateObj=None, action1=None, action2=None, bulk_input=None):
        basicValue = self.basicModel.getExpValue(stateHash, stateObj, action1, action2, bulk_input)
        trainedValue = self.trainedModel.getExpValue(stateHash, stateObj, action1, action2, bulk_input)

        if type(basicValue) == list:
            value = []
            for i in range(len(basicValue)):
                value.append([None if basicValue[i][0] == None else  basicValue[i][0] * (1-self.t) + trainedValue[i][0] * self.t])
        else:
            value = None if basicValue == None else basicValue * (1-self.t) + trainedValue * self.t

        if self.compare:
            if type(basicValue) == list:
                for i in range(len(basicValue)):
                    #None means basic has never seen it, so we have no good data
                    if basicValue[i][0] != None:
                        self.compPointsBasic.append(basicValue[i][0])
                        self.compPointsTrained.append(trainedValue[i][0])
            else:
                self.compPointsBasic.append(basicValue)
                self.compPointsTrained.append(trainedValue)

        return value

    def addReward(self, *args):
        self.basicModel.addReward(*args)
        self.trainedModel.addReward(*args)

    def train(self, epochs=1, batch_size=None):
        self.trainedModel.train(epochs, batch_size)

    def purge(self, seenStates):
        self.basicModel.purge(seenStates)
        self.trainedModel.purge(seenStates)

    def getMSE(self, clear=False):
        sum = 0
        count = 0
        for i in range(len(self.compPointsBasic)):
            b = self.compPointsBasic[i]
            t = self.compPointsTrained[i]
            sum += (b - t) ** 2
            count += 1
        if clear:
            self.compPointsBasic = []
            self.compPointsTrained = []
            self.compare = False

        if count == 0:
            return 0
        else:
            return sum / count

class TrainedModel:

    def __init__(self, alpha=0.001, model=None, width=256):
        self.alpha = alpha

        if model == None:
            #simple feedforward
            inputs = keras.Input(modelInput.inputShape)
            x = keras.layers.Dense(width, activation='relu')(inputs)
            y = keras.layers.Dense(width, activation='relu')(x)
            prediction = keras.layers.Dense(1, activation='sigmoid')(y)
            self.model = keras.Model(inputs=inputs, outputs=prediction)
            self._compile()
        else:
            self.model = model

        #used for training
        self.training = True
        self.savedInputs = []
        self.savedLabels = []

        self.expValueCache = {}

    def _compile(self):
        self.model.compile(
                optimizer=tf.train.AdamOptimizer(self.alpha),
                loss='logcosh')

    #uses the cached expValue if possible
    #otherwise generates it, adds it to cache
    def getExpValue(self, stateHash=None, stateObj=None, action1=None, action2=None, bulk_input=None):
        if (stateHash, action1, action2) in self.expValueCache:
            return self.expValueCache[(stateHash, action1, action2)]
        value = self.genExpValue(stateHash, stateObj, action1, action2)
        self.expValueCache[(stateHash, action1, action2)] = value
        return value


    #returns the expected value from the network
    def genExpValue(self, stateHash=None, stateObj=None, action1=None, action2=None, bulk_input=None):
        if bulk_input:
            data = [modelInput.toInput(so, a1, a2) for _, so, a1, a2 in bulk_input]
            return self.model.predict(np.array(data))
        else:
            data = modelInput.toInput(stateObj, action1, action2)
            return self.model.predict(np.array([data]))[0][0]

    #saves the data-label pair for training later
    def addReward(self, stateHash, stateObj, action1, action2, reward):
        if not self.training:
            return
        data = modelInput.toInput(stateObj, action1, action2)
        self.savedInputs.append(data)
        self.savedLabels.append(np.array([reward]))

    #trains on all the saved data-label pairs, then removing
    def train(self, epochs=1, batch_size=None):
        self.model.fit(np.array(self.savedInputs),
                np.array(self.savedLabels),
                verbose=0,
                epochs=epochs,
                batch_size=batch_size)
        self.savedInputs = []
        self.savedLabels = []
        self.expValueCache = {}

    #this doesn't need to purge, as memory usage doesn't grow much
    def purge(self, seenStates):
        pass

    #Save and load, also saves/loads the idMap from modeInput
    #dir should not include a trailing /
    def saveModel(self, dir, name):
        self.model.save(dir + '/' + name + '-model.h5')
        idMapData = pickle.dumps(modelInput.idMap)
        with open(dir + '/' + name + '-map.pickle', 'wb') as mapFile:
            mapFile.write(idMapData)

    def loadModel(self, dir, name):
        self.model = keras.model.load_model(dir + '/' + name + '-model.h5')
        self._compile()
        with open(dir + '/' + name + '-map.pickle', 'rb') as mapFile:
            idMapData = mapFile.read()
            modelInput.idMap = pickle.loads(idMapData)


class BasicModel:
    def __init__(self):
        self.rewardTable = collections.defaultdict(int)
        self.countTable = collections.defaultdict(int)
        #log holds a list of (stateHash, stateObj, action1, action2, reward) tuples
        #so these can be written out at some point an analyzed
        self.shouldLog = False
        self.log = []

    #returns the actual average reward for the (s,a,a) tuple
    def getExpValue(self, stateHash=None, stateObj=None, action1=None, action2=None, bulk_input=None):
        if bulk_input:
            #have to make this look like it came out of tf
            return [[self.getExpValue(*b, bulk_input=None)] for b in bulk_input]
        if self.shouldLog:
            self.log.append((stateHash, stateObj, action1, action2, reward))
        cumReward = self.rewardTable[(stateHash, action1, action2)]
        count = self.countTable[(stateHash, action1, action2)]
        return None if count == 0 else cumReward / count

    #adds the count and reward for the (s,a,a) tuple
    def addReward(self, stateHash, stateObj, action1, action2, reward):
        self.rewardTable[(stateHash, action1, action2)] += reward
        self.countTable[(stateHash, action1, action2)] += 1

    #removes information on states that haven't been seen
    def purge(self, seenStates):
        keys = list(self.rewardTable)
        for key in keys:
            stateHash = key[0]
            if not stateHash in seenStates:
                del self.rewardTable[key]
                del self.countTable[key]
