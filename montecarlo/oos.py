#!/usr/bin/env python3

import asyncio
import collections
import copy
import math
import numpy as np
import random
import sys

from game import Game
import moves

#Online Outcome Sampling
#http://mlanctot.info/files/papers/aij-2psimmove.pdf
#page 33

async def mcOOSImpl(requestQueue, cmdQueue, cmdHeader, mcData,
        format, playerNum, iter, initActions, pid=0, verbose=False):

    regretTable = mcData['regretTable']
    seenStates = mcData['seenStates']
    gamma = mcData['gamma'](iter)
    probTable = mcData['probTable']

    #stack where our actions/strategies are stored
    history = []

    running = True
    inInitActions = True
    while running:
        request = await requestQueue.get()
        if verbose:
            print('got request', cmdHeader, request)

        if request[0] == Game.REQUEST:
            req = request[1]
            state = req['stateHash']

            seenStates[state] = True
            actions = moves.getMoves(format, req)

            #after doing init actions, we're in the target state
            #need to reset the PRNG so the bot doesn't cheat
            if inInitActions and len(initActions) == 0:
                inInitActions = False
                await cmdQueue.put('>resetPRNG')

            #generate a stategy
            rSum = 0
            regrets = []
            for action in actions:
                regret = regretTable[(state, action)]
                regrets.append(regret)
                rSum += max(0, regret)
            if rSum > 0:
                #prob according to regret
                probs = np.array([max(0,r) / rSum for r in regrets])
                probs = probs / np.sum(probs)
                #use probs to update strategy
                #use exploreProbs to sample moves
                if iter % 2 == playerNum:
                    exploreProbs = probs * (1-gamma) + gamma / len(actions)
                else:
                    #we're the off player, don't explore
                    exploreProbs = probs
            else:
                #everything is new/bad, play randomly
                probs = np.array([1 / len(actions) for a in actions])
                exploreProbs = probs

            if len(initActions) > 0:
                #blindly pick init action
                preAction = initActions[0].strip()
                #find close enough action in list
                #PS client will generate team preview actions that
                #are longer than what we expect, but we can just
                #assume that the equivalent action is a prefix
                bestActionIndex = 0
                while bestActionIndex < len(actions):
                    if preAction.startswith(actions[bestActionIndex].strip()):
                        break
                    bestActionIndex += 1
                bestAction = actions[bestActionIndex]
                initActions = initActions[1:]
            else:
                #pick action based on probs
                bestActionIndex = np.random.choice(len(actions), p=exploreProbs)
                bestAction = actions[bestActionIndex]

            #save our action
            history.append((state, bestActionIndex, actions, probs, exploreProbs))

            if verbose:
                print('picked', cmdHeader + bestAction)

            await cmdQueue.put(cmdHeader + bestAction)

        elif request[0] == Game.END:
            running = False
            #map from [-1,1] to [0,1]
            reward = (request[1] + 1) / 2

            #on player's contribution to tail probability
            x = 1
            #on player's contribution to sample probability
            q = 1
            while len(history) > 0:
                state, actionIndex, actions, probs, exploreProbs = history.pop()
                if iter % 2 == playerNum:
                    action = actions[actionIndex]
                    w = reward * x / q
                    p = probs[actionIndex]
                    ep = exploreProbs[actionIndex]
                    #update picked action's regret
                    regretTable[(state, action)] += (1-p) / ep * w
                    #update other actions' regrets
                    for i in range(len(actions)):
                        if i == actionIndex:
                            continue
                        regretTable[(state, actions[i])] -= p / ep * w
                    x *= p
                    q *= ep
                else:
                    #update off player's average stategy
                    for i in range(len(actions)):
                        probTable[(state, actions[i])] += probs[i]





async def mcSearchOOS(ps, format, teams, mcData, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[],
        initExpVal=0, verbose=False):

    for i in range(len(mcData)):
        data = mcData[i]
        if not 'regretTable' in data:
            data['regretTable'] = collections.defaultdict(int)
        if not 'probTable' in data:
            data['probTable'] = collections.defaultdict(int)
        #gamma is actually epsilon in the paper
        #but we use gamma for this everywhere else
        if not 'gamma' in data:
            data['gamma'] = lambda iter: 0.3

        #always clear seen states
        data['seenStates'] = {}

    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=verbose)
        await game.startGame()
        await asyncio.gather(
                mcOOSImpl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0],
                    format=format, iter=i, playerNum=0,
                    initActions=p1InitActions,
                    verbose=verbose),
                mcOOSImpl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1],
                    format=format, iter=i, playerNum=1,
                    initActions=p2InitActions,
                    verbose=verbose))

    print(file=sys.stderr)



def getProbsOOS(mcData, state, actions):
    probTable = mcData['probTable']
    probs = np.array([probTable[(state, action)] for action in actions])
    return probs / np.sum(probs)

def combineOOSData(mcDatasets, valueModel=None):
    num = len(mcDatasets)
    #we'll assume that parallelization can be done naively
    #record which states were seen in the last iteration
    seenStates = {}
    for data in mcDatasets:
        for j in range(2):
            seen = data[j]['seenStates']
            for state in seen:
                seenStates[state] = True

    if valueModel:
        valueModel.purge(seenStates)

    print(len(seenStates))
    if num == 1:
        #no need to copy data around, just delete it directly
        data = mcDatasets[0]
        for j in range(2):
            probTable = data[j]['probTable']
            regretTable = data[j]['regretTable']
            keys = list(probTable)
            for state, action in keys:
                if state not in seenStates:
                    del probTable[(state, action)]
                    del regretTable[(state, action)]

        return mcDatasets
