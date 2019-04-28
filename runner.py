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

import config
import deepcfr
import model
import trashModel


#This file has functions relating to running the AI

async def trainAndPlay(file=sys.stdout):
    history = config.GameConfig.history

    m = mp.Manager()
    writeLock = m.Lock()
    if config.numProcesses > 0:
        trainingBarrier = m.Barrier(config.numProcesses)
    else:
        #barrier with 0 doesn't work
        trainingBarrier = m.Barrier(1)

    sharedDict = m.dict()

    #advModels = [model.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]
    stratModels = [trashModel.DeepCfrModel(name='strat' + str(i), softmax=True, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]
    advModels = [trashModel.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]

    for i in range(2):
        advModels[i].shareMemory()
        stratModels[i].shareMemory()

    agent = deepcfr.DeepCfrAgent(
            writeLock=writeLock,
            trainingBarrier=trainingBarrier,
            sharedDict=sharedDict,
            advModels=advModels,
            stratModels=stratModels,
            singleDeep=config.singleDeep,
            verbose=config.verboseTraining)

    #for debugging, as multiprocessing makes debugging difficult
    if config.numProcesses == 0:
        async with config.game.getContext() as context:
            await agent.search(
                context=context,
                pid=0,
                limit=config.limit,
                innerLoops=config.innerLoops,
                seed=config.seed,
                history=history)

    #instead of searching per turn, do all searching ahead of time
    processes = []
    for j in range(config.numProcesses):
        def run():
            async def asyncRun():
                async with config.game.getContext() as context:
                    await agent.search(
                        context=context,
                        pid=j,
                        limit=config.limit,
                        innerLoops=config.innerLoops,
                        seed=config.seed,
                        history=history)

            policy = asyncio.get_event_loop_policy()
            policy.set_event_loop(policy.new_event_loop())
            loop = asyncio.get_event_loop()
            loop.run_until_complete(asyncRun())

        if config.numProcesses > 0:
            p = mp.Process(target=run)
            p.start()
            processes.append(p)
            
    for p in processes:
        p.join()

    #we could have the agent do this when it's done training,
    #but I don't like having the agent worry about its own synchronization
    agent.stratTrain()
    print('final old model weights?', agent.oldModelWeights)

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

                    probs = agent.getProbs(player, infoset, actions)

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

                    await game.takeAction(player, req, action)

                await playTurn()


        #we're not searching, so additional games are free
        for i in range(config.numTestGames):
            seed = config.game.getSeed()            
            game = config.game.Game(context=context, seed=seed, history=history, verbose=True, file=file)
            await game.startGame()
            gameTask = asyncio.ensure_future(play(game))
            winner = await game.winner
            gameTask.cancel()
            print('winner:', winner, file=sys.stderr)
            print('|' + ('-' * 79), file=file)

