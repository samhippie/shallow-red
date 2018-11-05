#!/usr/bin/env python3

import asyncio
import collections
import copy
import math
import numpy as np
import random
import sys

from game import Game
import model
import moves

#MCCFR with External Sampling (might be Average Sampling later)

class CfrAgent:
    def __init__(self, teams, format, verbose=False):
        self. teams = teams
        self.format = format
        self.verbose = verbose

        self.regretTables = [collections.defaultdict(int) for i in range(2)]
        self.probTables = [collections.defaultdict(int) for i in range(2)]

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[], []]):
        #turn init actions into a useful history
        history = [(seed, a1, a2) for a1, a2 in zip(*initActions)]

        print(end='', file=sys.stderr)
        for i in range(limit):
            print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
            game = Game(ps, self.teams, format=self.format, seed=seed, verbose=self.verbose)
            await game.startGame()
            await game.applyHistory(history)
            await self.cfrRecur(ps, game, seed, history, 1, i)
        print(file=sys.stderr)

    def combine(self):
        #TODO see how bad this gets
        pass

    def getProbs(self, player, state, actions):
        pt = self.probTables[player]
        rt = self.probTables[player]
        probs = np.array([pt[(state, a)] for a in actions])
        rt = self.probTables[player]
        for a in actions:
            print('player', player, 'action', a, 'regret', rt[(state, a)])

        return probs / np.sum(probs)

    #recursive implementation of cfr
    #history is a list of (seed, action, action) tuples
    #q is the sample probability
    #assumes the game has already had the history applied
    async def cfrRecur(self, ps, game, startSeed, history, q, iter):
        async def endGame():
            side = 'bot1' if iter % 2 == 0 else 'bot2'
            winner = await game.winner
            #have to clear the results out of the queues
            while not game.p1Queue.empty():
                await game.p1Queue.get()
            while not game.p2Queue.empty():
                await game.p2Queue.get()
            if winner == side:
                return 1# / q
            else:
                return 0

        cmdHeaders = ['>p1', '>p2']
        queues = [game.p1Queue, game.p2Queue]
        offPlayer = (iter+1) % 2
        onPlayer = iter % 2

        #off player
        request = (await queues[offPlayer].get())
        if request[0] == Game.END:
            return await endGame()
        req = request[1]
        state = req['stateHash']
        actions = moves.getMoves(self.format, req)
        #just sample a move
        probs = self.regretMatch(offPlayer, state, actions)
        offAction = np.random.choice(actions, p=probs)
        #and update average stategy
        self.updateProbs(offPlayer, state, actions, probs)# / q)

        #on player
        request = (await queues[onPlayer].get())
        if request[0] == Game.END:
            return await endGame()
        req = request[1]
        state = req['stateHash']
        actions = moves.getMoves(self.format, req)
        #TODO make this part average sampling
        probs = self.regretMatch(onPlayer, state, actions)
        #get expected reward for each action
        rewards = []
        gameUsed = False
        for action, prob in zip(actions, probs):
            #don't have to re-init game for the first action
            if gameUsed:
                game = Game(ps, self.teams, format=self.format, seed=startSeed, verbose=self.verbose)
                await game.startGame()
                await game.applyHistory(history)
                #need to eat two requests, as we got two above
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

            r = await self.cfrRecur(ps, game, startSeed, history + [historyEntry], q * min(1, max(0.01, prob)), iter)
            rewards.append(r)

        #update regrets
        stateExpValue = 0
        for p,r in zip(probs, rewards):
            stateExpValue += p * r
        rt = self.regretTables[onPlayer]
        for a,r in zip(actions, rewards):
            rt[(state, a)] += r - stateExpValue

        return stateExpValue

    #generates probabilities for each action
    def regretMatch(self, player, state, actions):
        rt = self.regretTables[player]
        rSum = 0
        regrets = np.array([max(0, rt[(state, a)]) for a in actions])
        rSum = np.sum(regrets)
        return regrets / rSum if rSum > 0 else np.array([1/len(actions) for a in actions])

    #updates the average strategy for the player
    def updateProbs(self, player, state, actions, probs):
        pt = self.probTables[player]
        for a, p in zip(actions, probs):
            pt[(state, a)] += p
