
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
#whether to print out each line in our training games (for debugging)
verboseTraining = False

#data storage
dataDir = '/home/sam/data-ssd/'
#whether to store all samples in RAM rather than on disk
inMemory = False
#whether to explicitly cache samples from disk in RAM
bigCache = False

#playing actual non-training games
#ignore moves with probabilities below this (likely just noise)
probCutoff = 0.03
#how many games to play after training
numTestGames = 20

#general game config
gameName = 'warPoker'

if gameName == 'warPoker':
    game = games.warPoker
    GameConfig = WarPoker

    #search
    #number of search processes (limited by cpu utilization and gpu memory)
    numProcesses = 8
    #number of search iterations
    limit = 100
    #seed for all search games, None for default
    seed = None
    #which search iteration to start from, None for fresh start (delete data)
    resumeIter = 100
    #number of game tree traversals per search iteration
    innerLoops = 40
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still probed via rollout)
    branchingLimit = None
    #maximum depth in a traversal before rollout
    depthLimit = None

    #training
    #number of epochs for training the advantage network
    advEpochs = 40
    #number of epochs for training the strategy network
    stratEpochs = 60
    #number of samples in a batch
    miniBatchSize = 4
    #number of workers for the data loader
    numWorkers = 4
    #whether to create a fresh advantage network for each iteration
    newIterNets = True

    #model
    #number of bits for numbers in infosets
    numTokenBits = 5
    #maximum size for infoset vocabulary
    vocabSize = 64
    #size of embedding vector
    embedSize = 5
    embedDropoutPercent = 0.5
    #size of hidden state of lstm
    lstmSize = 32
    #number of lstm layers
    numLstmLayers = 2
    #size of each fully connected layer
    width = 32
    #learn rate for training
    learnRate = 0.004
    #how many samples to cache before writing to disk
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
