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
import model
import modelInput
import montecarlo.exp3 as exp3
import montecarlo.rm as rm
import montecarlo.oos as oos
import montecarlo.cfr as cfr

#This file has functions relating to running the AI

#putting this up top for convenience
def getAgent(algo, teams, format, valueModel=None):
    if algo == 'rm':
        agent = rm.RegretMatchAgent(
                teams=teams,
                format=format,
                posReg=True,
                probScaling=2,
                regScaling=1.5,
                tableType=rm.MEMORY,
                #dbLocation='/home/sam/scratch/psbot/rm-agent.db',
                verbose=False)
    elif algo == 'oos':
        agent = oos.OnlineOutcomeSamplingAgent(
                teams=teams,
                format=format,
                #posReg=True,
                #probScaling=2,
                #regScaling=1.5,
                verbose=False)
    elif algo == 'exp3':
        agent = exp3.Exp3Agent(
                teams=teams,
                format=format,
                verbose=False)
    elif algo == 'cfr':
        agent = cfr.CfrAgent(
                teams=teams,
                format=format,

                samplingType=cfr.EXTERNAL,
                exploration=0.1,
                bonus=0,
                threshold=1,
                bound=3,

                posReg=True,
                probScaling=2,
                regScaling=1.5,

                depthLimit=3,
                evaluation=cfr.ROLLOUT,

                verbose=False)
    return agent



#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'


async def playRandomGame(teams, format, ps=None, initMoves=[[],[]], seed=None):
    if not ps:
        ps = await getPSProcess()
    if not seed:
        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

    game = Game(ps, format=format, teams=teams, seed=seed, verbose=True)

    async def randomAgent(queue, cmdHeader, initMoveList):
        while True:
            req = await queue.get()
            if req[0] == Game.END:
                break

            #print('getting actions')
            actions = moves.getMoves(format, req[1])
            state = req[1]['state']
            #print(cmdHeader, 'actions', actions)

            if len(initMoveList) > 0:
                action = initMoveList[0]
                del initMoveList[0]
            else:
                action = random.choice(actions)
            print(cmdHeader, 'picked', action)
            await game.cmdQueue.put(cmdHeader + action)

    await game.startGame()
    gameTask = asyncio.gather(randomAgent(game.p1Queue, '>p1', initMoves[0]),
            randomAgent(game.p2Queue, '>p2', initMoves[1]))
    asyncio.ensure_future(gameTask)
    winner = await game.winner
    gameTask.cancel()
    print('winner:', winner)
    return winner


#plays two separately trained agents
#I just copied and pasted playTestGame and duplicated all the search functions
#so expect this to take twice as long as playTestGame with the same parameters
async def playCompGame(teams, limit1=100, limit2=100, time1=None, time2=None, format='1v1', numProcesses1=1, numProcesses2=1, algo1='rm', algo2='rm', file=sys.stdout, initMoves=([],[]), valueModel1=None, valueModel2=None, concurrent=False):
    try:

        mainPs = await getPSProcess()

        searchPs1 = [await getPSProcess() for i in range(numProcesses1)]
        searchPs2 = [await getPSProcess() for i in range(numProcesses2)]

        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

        if time1:
            limit1 = 100000000
        if time2:
            limit2 = 100000000

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        agent1 = getAgent(algo1, teams, format, valueModel1)
        agent2 = getAgent(algo2, teams, format, valueModel2)

        await game.startGame()

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            nonlocal searchPs1
            nonlocal searchPs2
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)

                #don't search if we aren't going to use the results
                if len(initMoves[0]) == 0 or len(initMoves[1]) == 0:

                    searches1 = []
                    searches2 = []
                    for j in range(numProcesses1):
                        search1 = agent1.search(
                                ps=searchPs1[j],
                                pid=j,
                                limit=limit1,
                                seed=seed)
                        searches1.append(search1)

                    for j in range(numProcesses2):
                        search2 = agent2.search(
                                ps=searchPs2[j],
                                pid=j,
                                limit=limit2,
                                seed=seed)
                        searches2.append(search2)

                    if concurrent and time1 == None and time2 == None:
                        await asyncio.gather(*searches1, *searches2)
                    elif time1 and time2:
                        #there's a timeout function, but I got this working first
                        searchTask = asyncio.ensure_future(asyncio.gather(*searches1))
                        await asyncio.sleep(time1)
                        searchTask.cancel()
                        with suppress(asyncio.CancelledError):
                            await searchTask

                        searchTask = asyncio.ensure_future(asyncio.gather(*searches2))
                        await asyncio.sleep(time2)
                        searchTask.cancel()
                        with suppress(asyncio.CancelledError):
                            await searchTask

                        #restart the search processes just to clean things up
                        #for ps in searchPs1 + searchPs2:
                            #ps.terminate
                        #searchPs1 = [await getPSProcess() for i in range(numProcesses1)]
                        #searchPs2 = [await getPSProcess() for i in range(numProcesses2)]

                    else:
                        await asyncio.gather(*searches1)
                        await asyncio.gather(*searches2)

                    #restart the search processes just to clean things up
                    for ps in searchPs1 + searchPs2:
                        ps.terminate()
                    searchPs1 = [await getPSProcess() for i in range(numProcesses1)]
                    searchPs2 = [await getPSProcess() for i in range(numProcesses2)]

                    #let the agents combine and purge data
                    print('combining', file=sys.stderr)
                    agent1.combine()
                    agent2.combine()


                #player-specific
                queues = [game.p1Queue, game.p2Queue]
                actionLists = [p1Actions, p2Actions]
                cmdHeaders = ['>p1', '>p2']
                agents = [agent1, agent2]

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
                        state = request[1]['stateHash']
                        actions = moves.getMoves(format, request[1])

                        probs = agents[num].getProbs(num, state, actions)
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

        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)
        return winner

    except:
        raise

    finally:
        mainPs.terminate()
        for ps in searchPs1:
            ps.terminate()
        for ps in searchPs2:
            ps.terminate()


#trains a model for the given format with the given teams
#returns the trained model
async def trainModel(teams, format, games=100, epochs=100, numProcesses=1, valueModel=None, saveDir=None, saveName=None):
    try:
        searchPs = [await getPSProcess() for p in range(numProcesses)]

        if not valueModel:
            valueModel = model.TrainedModel(alpha=0.0001)

        agent = getAgent('rm', teams, format, valueModel=valueModel)

        print('starting network training', file=sys.stderr)
        for i in range(epochs):
            print('epoch', i, 'running', file=sys.stderr)
            valueModel.t = i / epochs

            searches = []
            for j in range(numProcesses):
                search = agent.search(
                    ps=searchPs[j],
                    pid=j,
                    limit=games)
                searches.append(search)
            await asyncio.gather(*searches)

            print('epoch', i, 'training', file=sys.stderr)
            valueModel.train(epochs=10)
            if saveDir and saveName:
                valueModel.saveModel(saveDir, saveName)

    except:
        raise

    finally:
        for ps in searchPs:
            ps.terminate()
    return valueModel


async def playTestGame(teams, limit=100, time=None,
        format='1v1', seed=None, initMoves=([],[]),
        numProcesses=1,
        valueModel=None, algo='rm',
        #set bootstrap algo to start training with the bootstrap algorithm, then switch to the main algorithm
        #right now this only supports bootstrapping with RM and switching to CFR
        #(which is the only real use case)
        #also we're only supporting bootstrapping for game-limited searches, not time
        bootstrapAlgo=None, bootstrapPercentage=10,
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

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        agent = getAgent(algo, teams, format, valueModel)

        if bootstrapAlgo:
            bootAgent = getAgent(bootstrapAlgo, teams, format)

        if time:
            limit = 100000

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        await game.startGame()

        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)

                #don't search if we aren't going to use the results
                if len(initMoves[0]) == 0 or len(initMoves[1]) == 0:

                    #this is a bit messy, but we're just testing so it's
                    #"temporary"
                    if bootstrapAlgo:
                        bootLimit = int(limit * bootstrapPercentage / 100)
                        searches = []
                        for j in range(numProcesses):
                            search = bootAgent.search(
                                    ps=searchPs[j],
                                    pid=j,
                                    limit=bootLimit,
                                    seed=seed,
                                    initActions=[p1Actions, p2Actions])
                            searches.append(search)

                        await asyncio.gather(*searches)

                        agent.copyFromAgent(bootAgent)
                        bootAgent.combine()

                    searches = []
                    for j in range(numProcesses):
                        search = agent.search(
                                ps=searchPs[j],
                                pid=j,
                                limit=limit,
                                seed=seed,
                                initActions=[p1Actions, p2Actions])
                        searches.append(search)


                    #there's a timeout function, but I got this working first
                    if time:
                        searchTask = asyncio.ensure_future(asyncio.gather(*searches))
                        await asyncio.sleep(time)
                        searchTask.cancel()
                        with suppress(asyncio.CancelledError):
                            await searchTask

                    else:
                        await asyncio.gather(*searches)


                    #let the agents combine and purge data
                    print('combining', file=sys.stderr)
                    agent.combine()

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
                        state = request[1]['stateHash']
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


        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

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

