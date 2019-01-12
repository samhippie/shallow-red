
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
    limit = 50
    #seed for all search games, None for default
    seed = None
    #which search iteration to start from, None for fresh start (delete data)
    resumeIter = None
    #number of game tree traversals per search iteration
    innerLoops = 500
    #limit on number of branches to take per action in a traversal
    #(branches not taken are still probed via rollout)
    branchingLimit = None
    #maximum depth in a traversal before rollout
    depthLimit = None

    #training
    #number of epochs for training the advantage network
    advEpochs = 5
    #number of epochs for training the strategy network
    stratEpochs = 5
    #number of samples in a batch
    miniBatchSize = 16
    #number of workers for the data loader
    numWorkers = 16
    #whether to create a fresh advantage network for each iteration
    newIterNets = False

    #model
    #number of bits for numbers in infosets
    numTokenBits = 5
    #maximum size for infoset vocabulary
    vocabSize = 32
    #size of embedding vector
    embedSize = 10
    #dropout rate after embedding during training
    embedDropoutPercent = 0.2
    #size of hidden state of lstm
    lstmSize = 128
    #number of lstm layers
    numLstmLayers = 2
    #size of each fully connected layer
    width = 64
    #learn rate for training
    learnRate = 0.004
    #how many samples to cache before writing to disk (give or take)
    sampleCacheSize = 1000
    #max size on number of samples (only supported for on-disk sample storage)
    maxNumSamples = {
        'adv0': 20000,
        'adv1': 20000,
        'strat0': None,
        'strat1': None,
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
