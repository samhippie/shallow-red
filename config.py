
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
numTestGames = 10

#general game config
gameName = 'warPoker'

if gameName == 'warPoker':
    game = games.warPoker
    GameConfig = WarPoker

    #search
    numProcesses = 8
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
    embedSize = 10
    lstmSize = 1024
    width = 256
    learnRate = 0.0001
    sampleCacheSize = 1000


elif gameName == 'pokemon':
    pass
    #TODO

