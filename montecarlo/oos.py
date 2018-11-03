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

class OnlineOutcomeSamplingAgent:
    def __init__(self, teams, format, gamma=0.3, posReg=False, probScaling=0, regScaling=0, verbose=False):
        self.teams = teams
        self.format = format
        self.posReg = posReg
        self.probScaling = probScaling
        self.regScaling = regScaling
        self.verbose = verbose


        self.mcData = []
        for i in range(2):
            data = {
                'gamma': gamma,
                'regretTable': collections.defaultdict(int),
                'probTable': collections.defaultdict(int),
                'seenStates': {},
            }
            self.mcData.append(data)

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[],[]]):
        await mcSearchOOS(
                ps,
                self.format,
                self.teams,
                self.mcData,
                limit=limit,
                seed=seed,
                p1InitActions=initActions[0],
                p2InitActions=initActions[1],
                posReg=self.posReg,
                probScaling=self.probScaling,
                regScaling=self.regScaling,
                verbose=self.verbose)

    def getProbs(self, player, state, actions):
        probTable = self.mcData[player]['probTable']
        probs = np.array([probTable[(state, action)] for action in actions])
        return probs / np.sum(probs)

    def combine(self):
        self.mcData = combineOOSData([self.mcData])[0]


async def mcOOSImpl(requestQueue, cmdQueue, cmdHeader, mcData,
        format, playerNum, iter, initActions, pid=0,
        posReg=False, probScaling=0, regScaling=0, verbose=False):

    regretTable = mcData['regretTable']
    seenStates = mcData['seenStates']
    gamma = mcData['gamma']
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
                    regret = regretTable[(state, action)]
                    if regScaling != 0:
                        regret *= ((iter+1)**regScaling) / ((iter+1)**regScaling + 1)
                    regretTable[(state, action)] = regret + (1-p) / ep * w
                    #update other actions' regrets
                    for i in range(len(actions)):
                        if i == actionIndex:
                            continue
                        regret = regretTable[(state, actions[i])]
                        if regScaling != 0:
                            regret *= ((iter+1)**regScaling) / ((iter+1)**regScaling + 1)
                        if posReg:
                            regretTable[(state, actions[i])] = max(0, regret - p / ep * w)
                        else:
                            regretTable[(state, actions[i])] = regret - p / ep * w
                    x *= p
                    q *= ep
                else:
                    #update off player's average stategy
                    probScale = ((iter+1) / (iter+2))**probScaling
                    for i in range(len(actions)):
                        oldProb = probTable[(state, actions[i])]
                        probTable[(state, actions[i])] = probScale * oldProb + probs[i]


async def mcSearchOOS(ps, format, teams, mcData, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[],
        posReg=False, probScaling=0, regScaling=0, verbose=False):

    #always clear seen states
    mcData[0]['seenStates'] = {}
    mcData[1]['seenStates'] = {}

    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=verbose)
        await game.startGame()
        await asyncio.gather(
                mcOOSImpl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0],
                    format=format, iter=i, playerNum=0,
                    initActions=p1InitActions,
                    posReg=posReg, probScaling=probScaling,
                    regScaling=regScaling,
                    verbose=verbose),
                mcOOSImpl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1],
                    format=format, iter=i, playerNum=1,
                    initActions=p2InitActions,
                    posReg=posReg, probScaling=probScaling,
                    regScaling=regScaling,
                    verbose=verbose))

    print(file=sys.stderr)

def combineOOSData(mcDatasets):
    num = len(mcDatasets)
    #we'll assume that parallelization can be done naively
    #record which states were seen in the last iteration
    seenStates = {}
    for data in mcDatasets:
        for j in range(2):
            seen = data[j]['seenStates']
            for state in seen:
                seenStates[state] = True

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
                    if (state, action) in regretTable:
                        del regretTable[(state, action)]

        return mcDatasets
