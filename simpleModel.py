
import collections
import math
import numpy as np

import config

#a simple tabular model
#I'm only planning on making this single-threaded as a proof of concept

#all dicts have the form
#(infoset repr, action repr) => x
#where x is whatever is stored

#turns infoset/action into a key for the hash table
def infosetToRepr(infoset):
    #return hash(tuple(infoset))
    return tuple(infoset)

class TabModel:
    def __init__(self):
        #cumulative regret/probs/whatever
        self.values = collections.defaultdict(int)

    def shareMemory(self):
        pass#we're only supporting single-threads

    #we receive the advantage of the action, which is basically the immediate regret
    #so this can be plugged in to the deep cfr agent without modification
    def addSample(self, infoset, label, iter):
        infosetRepr = infosetToRepr(infoset)
        for action, regret in label:
            i = config.game.enumAction(action)
            oldValue = self.values[(infosetRepr, i)]
            newValue = math.sqrt(iter - 1) * oldValue / math.sqrt(iter) + regret / math.sqrt(iter)
            #equivalent to weighting linearly by iter, this is just numerically stable
            #newValue = (oldValue * (iter - 1) / iter + regret)# / math.sqrt(iter)
            #newValue = (oldValue + iter * regret) / math.sqrt(iter)
            newValue = max(0, newValue)
            self.values[(infosetRepr, i)] = newValue

    def clearSampleCache(self):
        pass#we store everything in memory

    def close(self):
        pass#nothing to close

    def predict(self, infoset, trace=False):
        infosetRepr = infosetToRepr(infoset)
        return np.array([self.values[(infosetRepr, i)] for i in range(config.game.numActions)])

    def train(self, epochs=1):
        pass#nothing to train

