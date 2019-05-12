#!/usr/bin/env python3

import asyncio
import collections
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
import copy
import math
import numpy as np
import os
import random
import sys
import subprocess
import torch.multiprocessing as mp
import torch.distributed as dist 

import config
import dataStorage
import deepcfr
import model
import nethandler


#This file has functions relating to running the AI

async def trainAndPlay(file=sys.stdout):
    history = config.GameConfig.history

    if len(sys.argv) < 3:
        print('include num processes and pid', file=sys.stderr)
        quit()

    elif 'PYTHONHASHSEED' not in os.environ:
        print('error PYTHONHASHSEED not set', file=sys.stderr)
        quit()

    numProcesses = int(sys.argv[1])
    pid = int(sys.argv[2])

    m = mp.Manager()
    writeLock = m.Lock()
    #if config.numProcesses > 0:

    sharedDict = m.dict()

    #stratModels = [model.DeepCfrModel(name='strat' + str(i), softmax=True, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]
    stratModels = []
    #right now agents don't directly use the model for evaluation, but the use it to write samples
    #TODO separate out the sample-writing so we can get rid of this 'useNet' business
    advModels = [model.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict, useNet=(pid == 0)) for i in range(2)]
    #advModels = []

    #for i in range(2):
        #advModels[i].shareMemory()
        #stratModels[i].shareMemory()

    agent = deepcfr.DeepCfrAgent(
            writeLock=writeLock,
            sharedDict=sharedDict,
            advModels=advModels,
            stratModels=stratModels,
            singleDeep=config.singleDeep,
            verbose=config.verboseTraining)

    #if there's only one process, just assume we don't want to train
    if numProcesses > 1:

        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
        dist.init_process_group('gloo', rank=pid, world_size = numProcesses)

        agentGroup = dist.new_group(ranks=list(range(1, numProcesses)))

        if pid == 0:
            print('setting up net process', file=sys.stderr)
            if config.resumeIter is None:
                dataStorage.clearData()
            dist.barrier()
            nethandler.run(sharedDict, agent, numProcesses)
        else:
            print('setting up search process', pid, file=sys.stderr)
            dist.barrier()
            async with config.game.getContext() as context:
                await agent.search(
                    context=context,
                    pid=pid-1,
                    limit=config.limit,
                    innerLoops=config.innerLoops,
                    distGroup=agentGroup,
                    seed=config.seed,
                    history=history)
    else:
        #copy in the untrained model for testing
        #in the future we could optionally load in trained models from the disk
        agent.oldModels[0] = [agent.advModels[0]]
        agent.oldModels[1] = [agent.advModels[1]]

    if pid != 0:
        return

    print('pid 0 continuing')


    #we could have the agent do this when it's done training,
    #but I don't like having the agent worry about its own synchronization
    #advModels = [model.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]
    #agent needs some properly initialized adv models for final evaluation
    agent.advModels = advModels
    #agent.stratTrain()
    #print('final old model weights?', agent.oldModelWeights)

    async with config.game.getContext() as context:
        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play(game):
            i = 0
            #actions taken so far by in the actual game
            while True:
                i += 1

                async def playTurn():

                    player, req, actions = await game.getTurn()
                    infoset = game.getInfoset(player)

                    probs = agent.getProbs(player, infoset, actions, game.prevTrajectories[player])

                    #remove low probability moves, likely just noise
                    normProbs = np.array([p if p > config.probCutoff else 0 for p in probs])
                    normSum = np.sum(normProbs)
                    if normSum > 0:
                        normProbs = normProbs / np.sum(normProbs)
                    else:
                        normProbs = [1 / len(actions) for a in actions]

                    for j in range(len(actions)):
                        actionString = config.game.prettyPrintMove(actions[j], req)
                        if normProbs[j] > 0:
                            print('|c|p' + str(player+1) + '|Turn ' + str(i) + ' action:', actionString,
                                    'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                    action = np.random.choice(actions, p=normProbs)

                    await game.takeAction(player, action)

                await playTurn()


        #we're not searching, so additional games are free
        for i in range(config.numTestGames):
            seed = config.game.getSeed()            
            game = config.game.Game(context=context, seed=seed, history=history, saveTrajectories=True, verbose=True, file=file)
            await game.startGame()
            gameTask = asyncio.ensure_future(play(game))
            winner = await game.winner
            gameTask.cancel()
            print('winner:', winner, file=sys.stderr)
            print('|' + ('-' * 79), file=file)

