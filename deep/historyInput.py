import numpy as np

#for turning an in-progress PS log to a series of vectors
#which could eventually be used with an RNN to generate a history representation

def logToVectors(log):
    #TODO filter out obviously useless lines
    for line in log:
        vec = lineToVector(line)
        if vec != None:
            #TODO generator or 2d numpy matrix?
            yield vec

def lineToVector(line):
    return None
