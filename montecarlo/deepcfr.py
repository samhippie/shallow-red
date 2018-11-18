#!/usr/bin/env python3

import asyncio
import collections
import copy
import math
import numpy as np
import random
import sys

from game import Game
import modelInput
import moves

#Deep MCCFR

#based on this paper
#https://arxiv.org/pdf/1811.00164.pdf
#not sure if this will be viable

#this agent breaks some compatibility with the other agents
#which I think is fine as we generally do all of our searching ahead of time
#so we'll need new runner functions

class DeepCfrModel:

    #for advantages, the input is the state vector
    #and the output is a vector of each move's advantage
    #for strategies, the input is the state vector
    #and the output is a vector of each move's probability

    #so the inputs are exactly the same (modelInput.stateSize), and the outputs
    #are almost the same (modelInput.numActions)
    #strategy is softmaxed, advantage is not


    def __init__(self):
        self.dataSet = []
        self.labelSet = []
        self.iterSet = []

        #TODO init our network so that we initially output 0 for everything

    def addSample(self, data, label, iter):
        self.dataSet.append(modelInput.stateToTensor(data))

        #list of non-zero indices
        labelIndices = []
        #list of the non-zero label values
        labelValues = []
        for action, value in label:
            labelIndices.append(modelInput.enumAction(action))
            labelValues.append(value)

        self.labelSet.append((labelIndices, labelValues))

        self.iterSet.append(iter)

    def predict(self, data):
        #TODO
        return [random.random() for i in range(modelInput.numActions)]

    def train(self):
        #TODO
        pass


class DeepCfrAgent:
    #each player gets one of each model
    #advModels calculates advantages
    #stratModels calculates average strategy
    def __init__(self, teams, format, advModels=None, stratModels=None, verbose=False):
        self. teams = teams
        self.format = format

        if advModels:
            self.advModels = advModels
        else:
            self.advModels = [DeepCfrModel() for i in range(2)]

        if stratModels:
            self.stratModels = stratModels
        else:
            self.stratModels = [DeepCfrModel() for i in range(2)]

        self.verbose = verbose

        self.regretTables = [{}, {}]
        self.probTables = [{}, {}]

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[], []]):
        #turn init actions into a useful history
        history = [(None, a1, a2) for a1, a2 in zip(*initActions)]
        #insert the seed in the first turn
        if len(history) > 0:
            _, a1, a2 = history[0]
            history[0] = (seed, a1, a2)

        print(end='', file=sys.stderr)
        for i in range(limit):
            print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
            game = Game(ps, self.teams, format=self.format, seed=seed, verbose=self.verbose)
            await game.startGame()
            await game.applyHistory(history)
            await self.cfrRecur(ps, game, seed, history, 1, i)
        print(file=sys.stderr)

    #note that state must be the object, not the hash
    def getProbs(self, player, state, actions):
        sm = self.stratModels[player]
        stratProbs = sm.predict(state)
        actionNums = [modelInput.enumAction(a) for a in actions]
        probs = []
        for n in actionNums:
            probs.append(stratProbs[n])
        probs = np.array(probs)

        pSum = np.sum(probs)
        if pSum > 0:
            return probs / np.sum(probs)
        else:
            return np.array([1 / len(actions) for a in actions])

    #recursive implementation of cfr
    #history is a list of (seed, action, action) tuples
    #q is the sample probability
    #assumes the game has already had the history applied
    async def cfrRecur(self, ps, game, startSeed, history, iter, depth=0):
        async def endGame():
            side = 'bot1' if iter % 2 == 0 else 'bot2'
            winner = await game.winner
            #have to clear the results out of the queues
            while not game.p1Queue.empty():
                await game.p1Queue.get()
            while not game.p2Queue.empty():
                await game.p2Queue.get()
            #the deep cfr paper uses [-1,1] rather than [0,1] for u
            if winner == side:
                return 1
            else:
                return -1

        cmdHeaders = ['>p1', '>p2']
        queues = [game.p1Queue, game.p2Queue]
        offPlayer = (iter+1) % 2
        onPlayer = iter % 2

        #off player
        request = (await queues[offPlayer].get())
        if request[0] == Game.END:
            return await endGame()
        req = request[1]
        state = req['state']
        actions = moves.getMoves(self.format, req)
        #just sample a move
        probs = self.regretMatch(offPlayer, state, actions)
        offAction = np.random.choice(actions, p=probs)
        #and update average stategy
        self.updateProbs(offPlayer, state, actions, probs, iter // 2 + 1)

        #on player
        request = (await queues[onPlayer].get())
        if request[0] == Game.END:
            return await endGame()
        req = request[1]

        #we're going to be trying all actions
        state = req['state']
        actions = moves.getMoves(self.format, req)
        probs = self.regretMatch(onPlayer, state, actions)

        #get expected reward for each action
        rewards = []
        gameUsed = False

        for action in actions:
            #don't have to re-init game for the first action
            if gameUsed:
                game = Game(ps, self.teams, format=self.format, seed=startSeed, verbose=self.verbose)
                await game.startGame()
                await game.applyHistory(history)
                #need to consume two requests, as we consumed two above
                await game.p1Queue.get()
                await game.p2Queue.get()
            else:
                gameUsed = True

            seed = Game.getSeed()
            if onPlayer == 0:
                onHeader = '>p1'
                offHeader = '>p2'
                historyEntry = (seed, action, offAction)
            else:
                onHeader = '>p2'
                offHeader = '>p1'
                historyEntry = (seed, offAction, action)

            await game.cmdQueue.put('>resetPRNG ' + str(seed))
            await game.cmdQueue.put(onHeader + action)
            await game.cmdQueue.put(offHeader + offAction)

            r = await self.cfrRecur(ps, game, startSeed, history + [historyEntry], iter, depth=depth+1)
            rewards.append(r)

        #save sample of advantages
        stateExpValue = 0
        for p,r in zip(probs, rewards):
            stateExpValue += p * r
        advantages = [r - stateExpValue for r in rewards]

        am = self.advModels[onPlayer]
        am.addSample(state, zip(actions, advantages), iter // 2 + 1)

        return stateExpValue

    #generates probabilities for each action
    #based on modeled advantages
    def regretMatch(self, player, state, actions):
        am = self.advModels[player]
        advs = am.predict(state)
        actionNums = [modelInput.enumAction(a) for a in actions]
        probs = []
        for n in actionNums:
            probs.append(max(0, advs[n]))
        probs = np.array(probs)
        pSum = np.sum(probs)
        if pSum > 0:
            return probs / pSum
        else:
            #pick the best action with probability 1
            best = None
            for i in range(len(actionNums)):
                n = actionNums[i]
                if best == None or advs[n] > advs[actionNums[best]]:
                    best = i
            probs = [0 for a in actions]
            probs[best] = 1
            return probs

    #adds sample of current strategy
    def updateProbs(self, player, state, actions, probs, iter):
        sm = self.stratModels[player]
        sm.addSample(state, zip(actions, probs), iter)
