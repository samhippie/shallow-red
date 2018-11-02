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

#Exp3

class Exp3Agent:
    def __init__(self, teams, format, gamma=0.3, verbose=False):
        self.teams = teams
        self.format = format
        self.verbose = verbose

        self.mcData = []
        for i in range(2):
            data = {
                'gamma': gamma,
                'countTable': collections.defaultdict(int),
                'expValueTable': collections.defaultdict(int),
                'seenStates': {},
            }
            self.mcData.append(data)

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[],[]]):

        await mcSearchExp3(
                ps,
                self.format,
                self.teams,
                self.mcData,
                limit=limit,
                seed=seed,
                p1InitActions=initActions[0],
                p2InitActions=initActions[1],
                verbose=self.verbose)

    def getProbs(self, player, state, actions):
        return getProbsExp3(self.mcData[player], state, actions)

    def combine(self):
        self.mcData = combineExp3Data([self.mcData])[0]


#initActions is a list of initial actions that will be blindy taken
#mcData has countTable, which maps (state, action) to count
#mcData has expValueTable, which maps (stat, action) to an expected value
#both should be defaultdict to 0
#mcData has gamma, which is a number [0,1], prob of picking random move
#iter is the iteration number, which may be used to compute gamma
async def mcExp3Impl(requestQueue, cmdQueue, cmdHeader, mcData,
        format, iter=0, initActions=[], verbose=False):

    countTable = mcData['countTable']
    expValueTable = mcData['expValueTable']
    gamma = mcData['gamma']
    seenStates = mcData['seenStates']

    #history so we can update probTable
    history = []

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
    inInitActions = True
    while running:
        request = await requestQueue.get()

        if verbose:
            print(cmdHeader, 'got request', request)

        if request[0] == Game.REQUEST or request[0] == Game.ERROR:
            req = request[1]
            state = req['stateHash']

            seenStates[state] = True
            actions = moves.getMoves(format, req)

            #check if we ran out of initActions on the previous turn
            #if so, we need to change the PRNG
            if inInitActions and len(initActions) == 0:
                inInitActions = False
                #no problem if both players reset the PRNG
                await cmdQueue.put('>resetPRNG')

            #calculate a probability for each action
            #need the probs from the initActions so we can update,
            #so we always calculate this
            eta = gamma / len(actions)
            expValues = [expValueTable[(state,action)] for action in actions]
            maxExpValues = max(expValues)
            ws = [expValueTable[(state, action)] - maxExpValues for action in actions]
            xs = [math.exp(eta * w) for w in ws]
            xSum = np.sum(xs)
            probs = np.array([(1-gamma) * x / xSum + gamma / len(actions) for x in xs])
            #illegal moves might have a negative probability, which should just be 0
            probs = [p if p > 0 else 0 for p in probs]
            probs = probs / np.sum(probs)

            if len(initActions) > 0:
                #blindly pick init action
                bestAction = initActions[0]
                bestActionIndex = actions.index(bestAction)
                bestActionProb = probs[bestActionIndex]
                initActions = initActions[1:]
            else:
                #pick action based on probs
                bestActionIndex = np.random.choice(len(actions), p=probs)
                bestAction = actions[bestActionIndex]
                bestActionProb = probs[bestActionIndex]

            #save our action
            history.append((state, bestAction, bestActionProb))

            if verbose:
                print('picked', cmdHeader + bestAction)

            await cmdQueue.put(cmdHeader + bestAction)

        elif request[0] == Game.END:
            #update probTable with our history + result
            reward = request[1]
            #rescale reward from [-1,1] to [0,1]
            reward = (reward + 1) / 2
            for state, action, prob in history:
                countTable[(state, action)] += 1
                expValueTable[(state,action)] += reward / prob

            running = False


#Exp3
async def mcSearchExp3(ps, format, teams, mcData, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[], verbose=False):

    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=verbose)
        await game.startGame()
        await asyncio.gather(
                mcExp3Impl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0], format=format,
                    initActions=p1InitActions, verbose=verbose),
                mcExp3Impl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1], format=format,
                    initActions=p2InitActions, verbose=verbose))
    print(file=sys.stderr)

def combineExp3Data(mcDatasets, valueModel=None):
    num = len(mcDatasets)
    #record which states were seen in the last iteration
    seenStates = {}
    for data in mcDatasets:
        for j in range(2):
            seen = data[j]['seenStates']
            for state in seen:
                seenStates[state] = True

    if valueModel:
        valueModel.purge(seenStates)

    if num == 1:
        for data in mcDatasets:
            for j in range(2):
                countTable = data[j]['countTable']
                expValueTable = data[j]['expValueTable']
                keys = list(countTable)
                for state, action in keys:
                    if state not in seenStates:
                        del countTable[(state, action)]
                        if (state, action) in expValueTable:
                            del expValueTable[(state, action)]
        return mcDatasets


    #combine data on states that were seen in any search
    #in the last iteration
    combMcData = [{
        'countTable': collections.defaultdict(int),
        'expValueTable': collections.defaultdict(int),
        'seenStates': {},
        'avgGamma': mcDatasets[0][j]['avgGamma'],
        'gamma': mcDatasets[0][j]['gamma']} for j in range(2)]

    for data in mcDatasets:
        for j in range(2):
            countTable = data[j]['countTable']
            expValueTable = data[j]['expValueTable']
            for state, action in countTable:
                if state in seenStates:
                    combMcData[j]['countTable'][(state, action)] += countTable[(state, action)]
                    combMcData[j]['expValueTable'][(state, action)] += expValueTable[(state, action)]

    #copy the combined data back into the datasets
    return [copy.deepcopy(combMcData) for j in range(num)]

#returns the final probabilities of each action in the state
def getProbsExp3(mcData, state, actions):
    countTable = mcData['countTable']
    counts = [countTable[(state, action)] for action in actions]
    expValueTable = mcData['expValueTable']
    totalCount = np.sum(counts)

    #not sure if I should make this adjustment or not
    #experiments seem to show that it helps

    gamma = mcData['gamma']
    probs = np.array([max(0, c - gamma * totalCount / len(actions)) for c in counts])
    probs = probs / np.sum(probs)
    return probs

#should return the expected value for the state
#I'm assuming x * p / c gives the expected value of a move with the given probability
#and averaging that for all moves gives the expected value for the state
#I'm not sure about the math but the numbers seem to work out
def getExpValueExp3(mcData, state, actions, probs):
    countTable = mcData['countTable']
    counts = [countTable[(state, action)] for action in actions]
    expValueTable = mcData['expValueTable']
    xvs = []
    for i in range(len(actions)):
        action = actions[i]
        if counts[i] == 0:
            continue
        xv = expValueTable[(state, action)] * probs[i] / counts[i]
        xvs.append(xv)
    return np.mean(xvs)


