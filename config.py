
import sys

import games.warPoker
import games.pokemon

#This is the file for configuring everything
#there may be some bugs if the size of things in models aren't powers of 2 (or at least even) #you have been warned

#game-specific configuration
class Pokemon:
    #format = '1v1'
    format = 'challengecup1v1'
    history = [[],[]]

class WarPoker:
    history = [[],[]]

#general settings

#search
#whether to print out each line in our training games (for debugging)
verboseTraining = False
#whether to print details for each validation pass
verboseValidation = True
#show the gradient plot every so many epochs (None to disable)
gradPlotStride = 200

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
numTestGames = 10

#general game config
#gameName = 'warPoker'
gameName = 'pokemon'

if gameName == 'warPoker':
    game = games.warPoker
    GameConfig = WarPoker

    #search
    #number of search processes (limited by cpu utilization and gpu memory)
    #set to 0 to run search in main thread (mainly for debugging)
    #not sure this is being used anymore
    numProcesses = 10
    #number of search iterations
    limit = 100
    #seed for all search games, None for default
    seed = None
    #which search iteration to start from, None for fresh start (delete data)
    resumeIter = None
    #number of game tree traversals per search iteration
    innerLoops = 1000
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still possibly probed via rollout)
    branchingLimit = 1
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
    progressGamesToRecord = 10
    progressGamePath = 'progress/'

    #training
    #number of epochs for training the advantage network
    advEpochs = 500
    #number of epochs for training the strategy network
    stratEpochs = 5
    #maximum number of samples in an epoch
    epochMaxNumSamples = 50000
    #number of samples in a batch
    miniBatchSize = 4096
    #number of workers for the data loader
    numWorkers = 1
    #whether to create a fresh advantage network for each iteration
    newIterNets = True
    singleDeep = True

    #model
    #number of bits for numbers in infosets
    numTokenBits = 0
    #maximum size for infoset vocabulary
    vocabSize = 256
    #size of embedding vector
    embedSize = 16
    #dropout rate after embedding during training
    embedDropoutPercent = 0.5

    #cnn stuff
    enableCnn = False

    #size of hidden state of the lstm (split in half if we're using a bidirection lstm)
    lstmSize = 16
    #number of lstm layers
    numLstmLayers = 1
    #dropout percentage for the lstm
    lstmDropoutPercent = 0
    #size of each fully connected layer
    width = 16

    #enable an attention layer after the lstm
    enableAttention = False

    #learn rate for training
    learnRate = 0.001 
    #which optimizer to use (adam or sgd)
    optimizer = 'adam'
    #whether to use a scheduler for the learning rate
    useScheduler = False
    #the patience of the schedule (# of epochs before reducing learn rate)
    schedulerPatience = 20
    #what factor to use to reduce the learn rate
    schedulerFactor = 0.5
    #what fraction of samples to use for validation
    valSplit = 0.3
    #how many samples to cache before writing to disk (give or take)
    sampleCacheSize = 1000
    #max size on number of samples (only supported for on-disk sample storage)
    #not sure that this is being used anymore
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
    innerLoops = 50
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still possibly probed via rollout)
    branchingLimit = 1
    #whether to probe branches not taken
    enableProbingRollout=False
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
    advEpochs = 200
    #number of epochs for training the strategy network
    stratEpochs = 5
    #maximum number of samples in an epoch
    epochMaxNumSamples = 1000000
    #number of samples in a batch
    miniBatchSize = 512
    #number of workers for the data loader
    numWorkers = 8
    #whether to create a fresh advantage network for each iteration
    newIterNets = True
    singleDeep = True

    #model
    #number of bits for numbers in infosets
    numTokenBits = 3
    #maximum size for infoset vocabulary
    vocabSize = 4096
    #size of embedding vector
    embedSize = 128
    #dropout rate after embedding during training
    embedDropoutPercent = 0.5

    #cnn stuff
    enableCnn = False

    #size of hidden state of the lstm
    lstmSize = 128
    #number of lstm layers
    numLstmLayers = 1
    #dropout percentage for the lstm
    lstmDropoutPercent = 0
    #size of each fully connected layer
    width = 32

    #enable an attention later after the lstm
    enableAttention = False

    #learn rate for training
    learnRate = 0.001
    #which optimizer to use (adam or sgd)
    optimizer = 'adam'
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
    }
