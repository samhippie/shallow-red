
import sys

import games.warPoker
#import games.pokemon

#This is the file for configuring everything

#game-specific configuration
class Pokemon:
    format = '1v1'
    history = [[],[]]

class WarPoker:
    history = [[],[]]

#general settings

#search
verboseTraining = False

#data storage
dataDir = '/home/sam/data/'
inMemory = False
bigCache = False

#playing actual non-training games
probCutoff = 0.03
numTestGames = 20

#general game config
gameName = 'warPoker'

if gameName == 'warPoker':
    game = games.warPoker
    GameConfig = WarPoker

    #search
    numProcesses = 8
    limit = 0
    seed = None
    resumeIter = 53
    innerLoops = 200
    branchingLimit = None
    depthLimit = None

    #training
    advEpochs = 4
    stratEpochs = 8
    #right now we don't support mini batching
    miniBatchSize = 16
    numWorkers = 16

    #model
    numTokenBits = 5
    vocabSize = 100
    embedSize = 5
    lstmSize = 1024
    width = 64
    learnRate = 0.0001
    sampleCacheSize = 1000


elif gameName == 'pokemon':
    #TODO something other than just copying and pasting from war poker
    """
    #search
    numProcesses = 0
    limit = 50
    seed = None
    resumeIter = None
    innerLoops = 200
    branchingLimit = None
    depthLimit = None

    #training
    advEpochs = 1000
    stratEpochs = 10000
    miniBatchSize = 4
    numWorkers = 16

    #model
    numTokenBits = 5
    vocabSize = 100
    embedSize = 5
    lstmSize = 128
    width = 64
    learnRate = 0.01
    sampleCacheSize = 1000
    """
