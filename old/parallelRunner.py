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
import torch.multiprocessing as mp

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

def getSharedDicts(algo, manager):
    sharedDicts = manager.dict()
    if algo == 'rm':
       sharedDicts['regretTable0'] = manager.dict()
       sharedDicts['regretTable1'] = manager.dict()
       sharedDicts['probTable0'] = manager.dict()
       sharedDicts['probTable1'] = manager.dict()
       sharedDicts['rewardTable'] = manager.dict()
       sharedDicts['countTable'] = manager.dict()
       return sharedDicts
    else:
        raise Exception

#forces the agent to use the shared dict
def applySharedDicts(algo, sharedDicts, agent):
    if algo == 'rm':
        agent.regretTables = [sharedDicts['regretTable0'], sharedDicts['regretTable1']]
        agent.probTables = [sharedDicts['probTable0'], sharedDicts['probTable1']]
        agent.rewardTable = sharedDicts['rewardTable']
        agent.countTable = sharedDicts['countTable']

        #and we could implement shared memory for other algorithms here
    else:
        raise Exception

#adds agent's data to the shared data
#this probably isn't thread safe, so use protection
def writeToSharedDict(algo, sharedDicts, agent):
    for i in range(2):
        srt = sharedDicts['regretTable' + str(i)]
        rt = agent.regretTables[i]
        for (key, value) in rt.items():
            if not key in srt:
                srt[key] = value
            else:
                srt[key] += value
        spt = sharedDicts['probTable' + str(i)]
        pt = agent.probTables[i]
        for (key, value) in pt.items():
            if not key in spt:
                spt[key] = value
            else:
                spt[key] += value

    for (key, value) in agent.rewardTable.items():
        if not key in sharedDicts['rewardTable']:
            sharedDicts['rewardTable'][key] = value
        else:
            sharedDicts['rewardTable'][key] += value

    for (key, value) in agent.countTable.items():
        if not key in sharedDicts['countTable']:
            sharedDicts['countTable'][key] = value
        else:
            sharedDicts['countTable'][key] += value

#copies data from shared dict to agent
#this is read only, so it won't interfere with anything
#and if another process writes to sharedDict, the problems will be minor
#this might make memory blow up
def copyFromSharedDict(algo, sharedDicts, agent):
    for i in range(2):
        srt = sharedDicts['regretTable' + str(i)]
        rt = agent.regretTables[i]
        for (key, value) in srt.items():
            rt[key] = value
        spt = sharedDicts['probTable' + str(i)]
        pt = agent.probTables[i]
        for (key, value) in spt.items():
            pt[key] = value

    for (key, value) in sharedDicts['rewardTable'].items():
        agent.rewardTable[key] = value

    for (key, value) in sharedDicts['countTable'].items():
        agent.countTable[key] = value



#runs the search for the agent, converting everything to shared memory if needed
async def doSearch(algo, sharedDicts, writeLock, agent, pid, limit, seed, p1Actions, p2Actions):
    #right now only rm is implemented (as rm is the best performing)
    #and we're just going to worry about sharing in-memory data
    #(a db might actually be easier to implement, though)
    if algo != 'rm' or agent.tableType != rm.MEMORY:
        raise Exception

    try:
        applySharedDicts(algo, sharedDicts, agent)

        searchPs = await getPSProcess()
        await agent.search(
                ps=searchPs,
                pid=pid,
                limit=limit,
                seed=seed,
                initActions=[p1Actions, p2Actions])

        #combine our data with the shared dict
        #print('combining data', pid, file=sys.stderr)
        #writeLock.acquire()
        #writeToSharedDict(algo, sharedDicts, agent)
        #writeLock.release()
        #copyFromSharedDict(algo, sharedDicts, agent)
        #print('done combining data', pid, file=sys.stderr)

    finally:
        searchPs.terminate()


#multiprocessing doesn't like async functions, so this works around that
#make a new process with this as the target
def runAsync(target, args=()):
    policy = asyncio.get_event_loop_policy()
    policy.set_event_loop(policy.new_event_loop())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(target(*args))

async def playTestGame(teams, limit=100, time=None,
        format='1v1', seed=None, initMoves=([],[]),
        numProcesses=1,
        valueModel=None, algo='rm',
        file=sys.stdout):
    try:

        mainPs = await getPSProcess()

        if not seed:
            seed = [
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
            ]

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        agent = getAgent(algo, teams, format, valueModel)

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        await game.startGame()

        m = mp.Manager()
        sharedDicts = getSharedDicts(algo, m)
        writeLock = m.Lock()

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

                    searches = []
                    for j in range(numProcesses):
                        p = mp.Process(target=runAsync, args=(doSearch, 
                                (algo, sharedDicts, writeLock, agent, j, limit, seed, p1Actions, p2Actions)))
                        p.start()
                        searches.append(p)

                    for p in searches:
                        p.join()

                    #game agent needs references to search agent data
                    applySharedDicts(algo, sharedDicts, agent)

                    #let the agents combine and purge data
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
                #the end game message might take a couple seconds
                #with multiprocessing, we can't cancel our searches once they're started
                #so it's better to wait to see if it shows up before searching
                await asyncio.sleep(3)


        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

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

