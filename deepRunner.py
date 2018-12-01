#!/usr/bin/env python3

import asyncio
import collections
from contextlib import suppress
import copy
import math
import numpy as np
import os
import random
import sys
import subprocess

import moves
from game import Game
import montecarlo.deepcfr as deepcfr

#This file has functions relating to running the AI
#with the deep cfr configuration
#which requires a different process than the other agents

#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

async def playTestGame(teams, limit=100,
        format='1v1', seed=None, initMoves=([],[]),
        numProcesses=1, advEpochs=100, stratEpochs=1000, branchingLimit=2, depthLimit=None,
        file=sys.stdout):
    try:
        mainPs = await getPSProcess()

        searchPs = [await getPSProcess() for i in range(numProcesses)]

        if not seed:
            seed = [
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
            ]

        agent = deepcfr.DeepCfrAgent(teams, format, advEpochs=advEpochs, stratEpochs=stratEpochs, branchingLimit=branchingLimit, depthLimit=depthLimit, verbose=False)

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        #instead of searching per turn, do all searching ahead of time
        searches = []
        for j in range(numProcesses):
            search = agent.search(
                    ps=searchPs[j],
                    pid=j,
                    limit=limit,
                    seed=seed,
                    initActions=initMoves)
            searches.append(search)

        await asyncio.gather(*searches)

        #we could have the agent do this when it's done training,
        #but I don't like having the agent worry about its own synchronization
        agent.stratTrain()

        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play(initMoves):
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            while True:
                i += 1

                #player-specific
                queues = [game.p1Queue, game.p2Queue]
                actionLists = [p1Actions, p2Actions]
                cmdHeaders = ['>p1', '>p2']

                async def playTurn(num):

                    request = await queues[num].get()

                    if len(initMoves[num]) > 0:
                        #do the given action
                        action = initMoves[num][0]
                        del initMoves[num][0]
                        print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' pre-set action:', action, file=file)
                    else:
                        #let the agent pick the action
                        #figure out what kind of action we need
                        state = request[1]['state']
                        actions = moves.getMoves(format, request[1])

                        probs = agent.getProbs(num, state, actions)
                        #remove low probability moves, likely just noise
                        normProbs = np.array([p if p > probCutoff else 0 for p in probs])
                        normSum = np.sum(normProbs)
                        if normSum > 0:
                            normProbs = normProbs / np.sum(normProbs)
                        else:
                            normProbs = [1 / len(actions) for a in actions]

                        for j in range(len(actions)):
                            actionString = moves.prettyPrintMove(actions[j], request[1])
                            if normProbs[j] > 0:
                                print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' action:', actionString,
                                        'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                        action = np.random.choice(actions, p=normProbs)

                    actionLists[num].append(action)
                    await game.cmdQueue.put(cmdHeaders[num] + action)

                await playTurn(0)
                await playTurn(1)


        #we're not searching, so additional games are free
        for i in range(10):
            seed = [
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
            ]
            game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)
            await game.startGame()
            gameTask = asyncio.ensure_future(play(copy.deepcopy(initMoves)))
            winner = await game.winner
            gameTask.cancel()
            print('winner:', winner, file=sys.stderr)
            print('|' + ('-' * 79), file=file)

    except:
        raise

    finally:
        mainPs.terminate()
        for ps in searchPs:
            ps.terminate()
        #a little dirty, not all agents need to be closed
        if callable(getattr(agent, 'close', None)):
            agent.close()


async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

