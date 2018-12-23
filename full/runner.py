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

import full.game
from full.game import Game
import full.deepcfr as deepcfr

#This file has functions relating to running the AI

#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

async def playTestGame(limit=100,
        format='1v1', seed=None, initMoves=[],
        numProcesses=1, advEpochs=100, stratEpochs=1000, branchingLimit=2, depthLimit=None, resumeIter=None,
        file=sys.stdout):
    try:

        #searchPs = [await getPSProcess() for i in range(numProcesses)]

        if not seed:
            seed = [
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
            ]

        m = mp.Manager()
        writeLock = m.Lock()
        trainingBarrier = m.Barrier(numProcesses)
        sharedDict = m.dict()

        agent = deepcfr.DeepCfrAgent(
                format,
                advEpochs=advEpochs,
                stratEpochs=stratEpochs,
                branchingLimit=branchingLimit,
                depthLimit=depthLimit,
                resumeIter=resumeIter,
                writeLock=writeLock,
                trainingBarrier=trainingBarrier,
                sharedDict=sharedDict,
                verbose=False)

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        #for debugging, as multiprocessing makes debugging difficult
        if numProcesses == 0:
            ps = await getPSProcess()
            try:
                await agent.search(
                    ps=ps,
                    pid=0,
                    limit=limit,
                    seed=seed,
                    initActions=initMoves)
            finally:
                ps.terminate()

        #instead of searching per turn, do all searching ahead of time
        processes = []
        for j in range(numProcesses):
            def run():
                print('running', j)
                async def asyncRun():
                    ps = await getPSProcess()
                    try:
                        await agent.search(
                            ps=ps,
                            pid=j,
                            limit=limit,
                            seed=seed,
                            initActions=initMoves)
                    finally:
                        ps.terminate()

                policy = asyncio.get_event_loop_policy()
                policy.set_event_loop(policy.new_event_loop())
                loop = asyncio.get_event_loop()
                loop.run_until_complete(asyncRun())

            if numProcesses > 0:
                p = mp.Process(target=run)
                p.start()
                processes.append(p)
                
        for p in processes:
            p.join()

        #we could have the agent do this when it's done training,
        #but I don't like having the agent worry about its own synchronization
        agent.stratTrain()

        mainPs = await getPSProcess()

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
                    normProbs = np.array([p if p > probCutoff else 0 for p in probs])
                    normSum = np.sum(normProbs)
                    if normSum > 0:
                        normProbs = normProbs / np.sum(normProbs)
                    else:
                        normProbs = [1 / len(actions) for a in actions]

                    for j in range(len(actions)):
                        actionString = Game.prettyPrintMove(actions[j], request[1])
                        if normProbs[j] > 0:
                            print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' action:', actionString,
                                    'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                    action = np.random.choice(actions, p=normProbs)

                    await game.takeAction(player, req, action)

                await playTurn()


        #we're not searching, so additional games are free
        for i in range(10):
            seed = [
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
            ]
            game = Game(mainPs, format=format, teams=teams, seed=seed, history=initMoves, verbose=True, file=file)
            await game.startGame()
            gameTask = asyncio.ensure_future(play(game))
            winner = await game.winner
            gameTask.cancel()
            print('winner:', winner, file=sys.stderr)
            print('|' + ('-' * 79), file=file)

    except:
        raise

    finally:
        mainPs.terminate()
        #a little dirty, not all agents need to be closed
        if callable(getattr(agent, 'close', None)):
            agent.close()


async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

