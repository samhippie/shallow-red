
import sys

import games.warPoker
import games.pokemon

#This is the file for configuring everything
#there may be some bugs if the size of things in models aren't powers of 2 (or at least even) #you have been warned

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
numTestGames = 1000000

#general game config
#gameName = 'warPoker'
gameName = 'pokemon'

if gameName == 'warPoker':
    game = games.warPoker
    GameConfig = WarPoker

    #search
    #number of search processes (limited by cpu utilization and gpu memory)
    #set to 0 to run search in main thread (mainly for debugging)
    numProcesses = 10
    #number of search iterations
    limit = 100
    #seed for all search games, None for default
    seed = None
    #which search iteration to start from, None for fresh start (delete data)
    resumeIter = None
    #number of game tree traversals per search iteration
    innerLoops = 200
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still possibly probed via rollout)
    branchingLimit = None
    #whether to probe branches not taken
    enableProbingRollout=True
    #maximum depth in a traversal before rollout
    depthLimit = None
    #odds of the off player making a random move
    offExploreRate = 0
    #odds of the on player making a random move
    #only used if branchingLimit is not none
    onExploreRate = 0.2
    #how many games to record per training iteration
    progressGamesToRecord = 5
    progressGamePath = 'progress/'

    #training
    #number of epochs for training the advantage network
    advEpochs = 100
    #number of epochs for training the strategy network
    stratEpochs = 5
    #maximum number of samples in an epoch
    epochMaxNumSamples = 30000
    #number of samples in a batch
    miniBatchSize = 4096
    #number of workers for the data loader
    numWorkers = 8
    #whether to create a fresh advantage network for each iteration
    newIterNets = True
    singleDeep = True

    #model
    #number of bits for numbers in infosets
    numTokenBits = 0
    #maximum size for infoset vocabulary
    vocabSize = 128
    #size of embedding vector
    embedSize = 4
    #dropout rate after embedding during training
    embedDropoutPercent = 0.2

    #cnn stuff
    enableCnn = False
    #output size of convolutions
    convSizes = [8, 16]
    #kernel sizes of convolutions (should be odd)
    kernelSizes = [3, 5]
    #list of convolution layers and depths
    convDepths = [3, 3]
    #list of (approximate) pooling output sizes (which determines the kernel size)
    poolSizes = [8, 4]

    #size of hidden state of the lstm (split in half if we're using a bidirection lstm)
    lstmSize = 32
    #number of lstm layers
    numLstmLayers = 1
    #dropout percentage for the lstm
    lstmDropoutPercent = 0
    #size of each fully connected layer
    width = 8

    #enable an attention later after the lstm
    enableAttention = True

    #learn rate for training
    learnRate = 0.001 #whether to use a scheduler for the learning rate
    useScheduler = True
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
    game = games.pokemon
    GameConfig = Pokemon

    #search
    #number of search iterations
    limit = 300
    #seed for all search games, None for default
    seed = None
    #which search iteration to start from, None for fresh start (delete data)
    resumeIter = None
    #number of game tree traversals per search iteration
    innerLoops = 30
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still possibly probed via rollout)
    branchingLimit = 1
    #whether to probe branches not taken
    enableProbingRollout=True
    #maximum depth in a traversal before rollout
    depthLimit = 20
    #odds of the off player making a random move
    offExploreRate = 0
    #odds of the on player making a random move
    #only used if branchingLimit is not none
    onExploreRate = 0.2
    #how many games to record per training iteration
    progressGamesToRecord = 6
    progressGamePath = 'progress/'

    #training
    #number of epochs for training the advantage network
    advEpochs = 300
    #number of epochs for training the strategy network
    stratEpochs = 5
    #maximum number of samples in an epoch
    epochMaxNumSamples = 30000
    #number of samples in a batch
    miniBatchSize = 2048
    #number of workers for the data loader
    numWorkers = 8
    #whether to create a fresh advantage network for each iteration
    newIterNets = True
    singleDeep = True

    #model
    #number of bits for numbers in infosets
    numTokenBits = 10
    #maximum size for infoset vocabulary
    vocabSize = 4096
    #size of embedding vector
    embedSize = 8
    #dropout rate after embedding during training
    embedDropoutPercent = 0

    #cnn stuff
    enableCnn = False
    #output size of convolutions
    convSizes = [8, 16, 32]
    #kernel sizes of convolutions (should be odd)
    kernelSizes = [3, 5, 7]
    #list of convolution layers and depths
    convDepths = [3, 3, 3]
    #list of (approximate) pooling output sizes (which determines the kernel size)
    poolSizes = [128, 64, 16]

    #size of hidden state of the lstm
    lstmSize = 48
    #number of lstm layers
    numLstmLayers = 1
    #dropout percentage for the lstm
    lstmDropoutPercent = 0
    #size of each fully connected layer
    width = 32

    #enable an attention later after the lstm
    enableAttention = True

    #learn rate for training
    learnRate = 0.001
    #whether to use a scheduler for the learning rate
    useScheduler = True
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
    }
