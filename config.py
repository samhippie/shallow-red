
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
    #set to 0 to run search in main thread (mainly for debugging)
    numProcesses = 0
    #number of search iterations
    limit = 20
    #seed for all search games, None for default
    seed = None
    #which search iteration to start from, None for fresh start (delete data)
    resumeIter = None
    #number of game tree traversals per search iteration
    innerLoops = 100#0
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still probed via rollout)
    branchingLimit = None
    #maximum depth in a traversal before rollout
    depthLimit = None
    #odds of the off player making a random move
    exploreRate = 0

    #training
    #number of epochs for training the advantage network
    advEpochs = 100
    #number of epochs for training the strategy network
    stratEpochs = 5
    #maximum number of samples in an epoch
    epochMaxNumSamples = 10000
    #number of samples in a batch
    miniBatchSize = 1
    #number of workers for the data loader
    numWorkers = 4
    #whether to create a fresh advantage network for each iteration
    newIterNets = True
    singleDeep = True

    #model
    #number of bits for numbers in infosets
    numTokenBits = 0
    #maximum size for infoset vocabulary
    vocabSize = 128
    #size of embedding vector
    embedSize = 32
    #dropout rate after embedding during training
    embedDropoutPercent = 0
    #output size of convolutions
    convSizes = [16, 32]
    #kernel sizes of convolutions (should be odd)
    kernelSizes = [3, 5]
    #list of convolution layers and depths
    convDepths = [2, 2]
    #list of (approximate) pooling output sizes (which determines the kernel size)
    poolSizes = [13, 5]
    #size of hidden state of the lstm
    lstmSize = 16
    #number of lstm layers
    numLstmLayers = 1
    #dropout percentage for the lstm
    lstmDropoutPercent = 0
    #size of each fully connected layer
    width = 16

    #learn rate for training
    learnRate = 0.001
    #whether to use a scheduler for the learning rate
    useScheduler = False
    #the patience of the schedule (# of epochs before reducing learn rate)
    schedulerPatience = 10
    #what factor to use to reduce the learn rate
    schedulerFactor = 0.5
    #what fraction of samples to use for validation
    valSplit = 0.3
    #how many samples to cache before writing to disk (give or take)
    sampleCacheSize = 1000
    #max size on number of samples (only supported for on-disk sample storage)
    maxNumSamples = {
        'adv0': None,
        'adv1': None,
        'strat0': 50000,
        'strat1': 50000,
    }


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
