#!/usr/bin/env python3

import collections
import numpy as np
import os
import tensorflow as tf
from tensorflow import keras

import modelInput

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
        self.savedInputs = []
        self.savedLabels = []

    def _compile(self):
        self.model.compile(
                optimizer=tf.train.AdamOptimizer(self.alpha),
                loss='logcosh')

    #returns the expected value from the network
    def getExpValue(self, stateHash=None, stateObj=None, action1=None, action2=None, bulk_input=None):
        if bulk_input:
            data = [modelInput.toInput(so, a1, a2) for _, so, a1, a2 in bulk_input]
            return self.model.predict(np.array(data))
        else:
            data = modelInput.toInput(stateObj, action1, action2)
            return self.model.predict(np.array([data]))[0][0]

    #saves the data-label pair for training later
    def addReward(self, stateHash, stateObj, action1, action2, reward):
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

    #Save and load need to save/load the idMap from modeInput
    def saveModel(self, name):
        pass

    def loadModel(self, name):
        pass


class BasicModel:
    def __init__(self):
        self.rewardTable = collections.defaultdict(int)
        self.countTable = collections.defaultdict(int)

    #returns the actual average reward for the (s,a,a) tuple
    def getExpValue(self, stateHash=None, stateObj=None, action1=None, action2=None, bulk_input=None):
        if bulk_input:
            #have to make this look like it came out of tf
            return [[self.getExpValue(*b, bulk_input=None)] for b in bulk_input]
        cumReward = self.rewardTable[(stateHash, action1, action2)]
        count = self.countTable[(stateHash, action1, action2)]
        return None if count == 0 else cumReward / count

    #adds the count and reward for the (s,a,a) tuple
    def addReward(self, stateHash, stateObj, action1, action2, reward):
        self.rewardTable[(stateHash, action1, action2)] += reward
        self.countTable[(stateHash, action1, action2)] += 1
