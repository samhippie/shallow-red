#!/usr/bin/env python3

import asyncio
import collections
from concurrent.futures import ProcessPoolExecutor
from contextlib import suppress
import copy
import datetime
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

async def trainAndPlay(numProcesses, pid, saveFile=None, clear=False, file=sys.stdout):
    if saveFile and saveFile[-1] != '/':
        saveFile += '/'

    history = config.GameConfig.history

    if 'PYTHONHASHSEED' not in os.environ:
        print('error PYTHONHASHSEED not set', file=sys.stderr)
        quit()

    m = mp.Manager()
    writeLock = m.Lock()

    sharedDict = m.dict()

    oldModels = [[], []]
    oldModelWeights = [[], []]
    if pid == 0 and saveFile:
        if os.path.isdir(saveFile):
            for filename in os.listdir(saveFile):
                #in format "blah/model.adv(0|1).[n].pt"
                #adv0 or adv1 are which player
                #n is the iteration number
                parts = filename.split('.')
                if len(parts) < 4 or parts[-1] != 'pt':
                    continue
                #player
                if parts[-3] == 'adv0':
                    player = 0
                elif parts[-3] == 'adv1':
                    player = 1
                else:
                    continue
                #iteration
                n = int(parts[-2])
                oldModel = model.DeepCfrModel(name='adv' + str(player), softmax=False, writeLock=writeLock, sharedDict=sharedDict, useNet=True, saveFile=saveFile)
                oldModel.loadModel(n)
                oldModels[player].append(oldModel.net)
                oldModelWeights[player].append(n)
        else:
            os.mkdir(saveFile)

    stratModels = []
    #right now agents don't directly use the model for evaluation, but the use it to write samples
    #TODO separate out the sample-writing so we can get rid of this 'useNet' business
    advModels = [model.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict, useNet=(pid == 0), saveFile=saveFile) for i in range(2)]
    #advModels = []

    #for i in range(2):
        #advModels[i].shareMemory()
        #stratModels[i].shareMemory()

    if pid == 0:
        if len(oldModels[0]) > 0:
            weight, i = max([(weight, i) for i, weight in enumerate(oldModelWeights[0])])
            advModels[0].net = oldModels[0][i]
        if len(oldModels[1]) > 0:
            weight, i = max([(weight, i) for i, weight in enumerate(oldModelWeights[1])])
            advModels[1].net = oldModels[1][i]

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
        #3 hour timeout
        #shouldn't be an issue
        dist.init_process_group('gloo', timeout=datetime.timedelta(0, 10800000), rank=pid, world_size = numProcesses)

        agentGroup = dist.new_group(ranks=list(range(1, numProcesses)))

        if pid == 0:
            print('setting up net process', file=sys.stderr)
            if clear or saveFile is None:
                dataStorage.clearData()
            dist.barrier()
            agent.oldModels = oldModels
            agent.oldModelWeights = oldModelWeights
            await nethandler.run(agent, numProcesses, testGames, config.progressGamesToRecord)
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
    elif not saveFile:
        #copy in the untrained model for testing
        agent.oldModels = [[agent.advModels[0].net], [agent.advModels[1].net]]
        agent.oldModelWeights = [[1],[1]]
    elif saveFile and pid == 0:
        #assume that if there is a save file set, there are some models there
        agent.oldModels = oldModels
        agent.oldModelWeights = oldModelWeights

    if pid != 0:
        return

    print('pid 0 continuing')

    await testGames(agent, config.numTestGames, file)

async def testGames(agent, num, file=sys.stdout):
    history = config.GameConfig.history
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

                    probs = agent.getProbs(player, infoset, actions, game.prevTrajectories[player], file=file)

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

                    actionIndex = np.random.choice(len(actions), p=normProbs)

                    await game.takeAction(player, actionIndex)

                await playTurn()


        #we're not searching, so additional games are free
        for i in range(num):
            seed = config.game.getSeed()            
            game = config.game.Game(context=context, seed=seed, history=history, saveTrajectories=True, verbose=True, file=file)
            await game.startGame()
            gameTask = asyncio.ensure_future(play(game))
            winner = await game.winner
            gameTask.cancel()
            print('winner:', winner, file=file)
            print('|' + ('-' * 79), file=file)

