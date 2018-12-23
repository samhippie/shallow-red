#!/usr/bin/env python3

import asyncio
import collections
import copy
import math
import numpy as np
import random
import sys

import full.game
from full.game import Game
import full.model
import full.dataStorage
from full.action import actionMap

#Deep MCCFR

#based on this paper
#https://arxiv.org/pdf/1811.00164.pdf

#this agent breaks some compatibility with the other agents
#which I think is fine as we generally do all of our searching ahead of time
#so we'll need new runner functions

class DeepCfrAgent:
    #each player gets one of each model
    #advModels calculates advantages
    #stratModels calculates average strategy

    #branch limit is the maximum number of actions taken
    #actions that aren't taken are probed (i.e. rolled out)

    #depth limit is the maximum number of turns taken from the root
    #after the limit it hit, all games are evaluated via rollout
    #this agent is only applied at the root, so depth limit might
    #significantly affect the quality of late-game strategies
    #(although we could use RM to find new late-game strategies,
    #but that's outside the scope of this agent)

    #resumeIter is the iteration to resume from
    #this really should be a paremeter to the search function,
    #but we need to know about it when we initialize the models
    #this is a problem with our agent being made for a one-shot training cycle
    #instead of multiple training cycles like the others
    def __init__(
            self,
            format,
            writeLock,
            trainingBarrier,
            sharedDict,
            advModels=None, stratModels=None,
            advEpochs=1000,
            stratEpochs=10000,
            branchingLimit=None,
            depthLimit=None,
            resumeIter=None,
            verbose=False):

        self.format = format

        self.pid = -1

        self.resumeIter = resumeIter


        self.writeLock = writeLock
        self.trainingBarrier = trainingBarrier
        self.sharedDict = sharedDict

        if resumeIter == None:
            #fresh start, delete old data
            full.dataStorage.clearData()
            

        if advModels:
            self.advModels = advModels
        else:
            self.advModels = [full.model.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]

        if stratModels:
            self.stratModels = stratModels
        else:
            self.stratModels = [full.model.DeepCfrModel(name='strat' + str(i), softmax=True, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]

        self.advEpochs = advEpochs
        self.stratEpochs = stratEpochs

        self.branchingLimit = branchingLimit
        self.depthLimit = depthLimit

        self.verbose = verbose

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[]):
        self.pid = pid


        start = self.resumeIter if self.resumeIter else 0

        if self.pid == 0:
            print(end='', file=sys.stderr)
        for i in range(start, limit):
            if self.pid == 0:
                print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)

            #for self.small games, this is necessary to get a decent number of samples
            for j in range(1):
                #need idMap to be the same across processes

                game = Game(ps, format=self.format, seed=seed, history=initActions, verbose=self.verbose)
                await game.startGame()
                await self.cfrRecur(ps, game, seed, initActions, i)

            #save our adv data after each iteration
            #so the non-zero pid workers don't have data cached
            self.advModels[i % 2].clearSampleCache()

            #only need to train about once per iteration
            self.trainingBarrier.wait()
            if self.pid == 0:
                self.advTrain(i % 2)
                self.sharedDict['advNet' + str(i % 2)] = self.advModels[i % 2].net
            self.trainingBarrier.wait()
            #broadcast the new network back out
            self.advModels[i % 2].net = self.sharedDict['advNet' + str(i % 2)]

        #clear the sample caches so the master agent can train with our data
        for sm in self.stratModels:
            sm.clearSampleCache()

        self.trainingBarrier.wait()

        if self.pid == 0:
            print(file=sys.stderr)

    def advTrain(self, player):
        model = self.advModels[player]
        model.train(epochs=self.advEpochs)

    def stratTrain(self):
        if self.pid == 0:
            print('training strategy', file=sys.stderr)
        #we train both strat models at once
        for model in self.stratModels:
            model.train(epochs=self.stratEpochs)

    def getProbs(self, player, infoset, actions):
        sm = self.stratModels[player]
        stratProbs = sm.predict(infoset)
        actionNums = [actionMap[a] for a in actions]
        probs = []
        for n in actionNums:
            probs.append(stratProbs[n])
        probs = np.array(probs)

        pSum = np.sum(probs)
        if pSum > 0:
            return probs / np.sum(probs)
        else:
            return np.array([1 / len(actions) for a in actions])

    #closes the models, which have data to clean up
    def close(self):
        for model in self.advModels + self.stratModels:
            model.close()

    #recursive implementation of cfr
    #history is a list of (seed, player, action) tuples
    #assumes the game has already had the history applied
    async def cfrRecur(self, ps, game, startSeed, history, iter, depth=0, rollout=False):
        if depth > self.depthLimit:
            rollout = True

        onPlayer = iter % 2
        offPlayer = (iter + 1) % 2

        player, req, actions = await game.getTurn()

        if 'win' in req:
            if req['win'] and player == onPlayer:
                return 1
            else:
                return -1

        infoset = game.getInfoset(player)

        if player == offPlayer:
            #get probs so we can sample a single action
            probs = self.regretMatch(offPlayer, infoset, actions, -1)
            action = np.random.choice(actions, p=probs)
            #save sample for final average strategy
            if not rollout:
                self.updateProbs(offPlayer, infoset, actions, probs, iter // 2 + 1)

            if depth == 0 and self.pid == 0:
                print('player ' + str(player) + ' probs', list(zip(actions, probs)), file=sys.stderr)

            await game.takeAction(player, req, action)
            return await self.cfrRecur(ps, game, startSeed, history, iter, depth, rollout)

        elif player == onPlayer:
            #get probs, which action we take depends on the configuration
            probs = self.regretMatch(onPlayer, infoset, actions, depth)
            if depth == 0 and self.pid == 0:
                print('player ' + str(onPlayer) + ' probs', list(zip(actions, probs)), file=sys.stderr)
            if rollout:
                #we pick one action according to the current strategy
                actions = [np.random.choice(actions, p=probs)]
                actionIndices = [0]
            elif self.branchingLimit:
                #select a set of actions to pick
                #chance to play randomly instead of picking the best actions
                exploreProbs = probs# * (0.9) + 0.1 / len(probs)
                #there might be some duplicates but it shouldn't matter
                actionIndices = np.random.choice(len(actions), self.branchingLimit, p=exploreProbs)
            else:
                #we're picking every action
                actionIndices = list(range(len(actions)))

            #get expected reward for each action
            rewards = []
            gameUsed = False

            for i in range(len(actions)):
                action = actions[i]

                #use rollout for non-sampled actions
                if not i in actionIndices:
                    curRollout = True
                else:
                    curRollout = rollout

                #don't have to re-init game for the first action
                if gameUsed:
                    game = Game(ps, format=self.format, seed=startSeed, history=history, verbose=self.verbose)
                    await game.startGame()
                    await game.getTurn()
                else:
                    gameUsed = True

                #I want to see if we get good results by keeping the RNG the same
                #this is closer to normal external sampling
                #seed = await game.resetSeed()
                await game.takeAction(player, req, action)
                historyEntry = (None, player, action)

                r = await self.cfrRecur(ps, game, startSeed, history + [historyEntry], iter, depth=depth+1, rollout=curRollout)
                rewards.append(r)

            if not rollout:
                #save sample of advantages
                stateExpValue = 0
                for p,r in zip(probs, rewards):
                    stateExpValue += p * r
                advantages = [r - stateExpValue for r in rewards]

                am = self.advModels[onPlayer]
                am.addSample(infoset, zip(actions, advantages), iter // 2 + 1)

                if depth == 0 and self.pid == 0:
                    print('player', str(onPlayer), file=sys.stderr)
                    print('stateExpValue', stateExpValue, 'from', list(zip(probs, rewards)), file=sys.stderr)
                    print('advantages', list(zip(actions, advantages)), file=sys.stderr)

                return stateExpValue
            else:
                #we can't calculate advantage, so we can't update anything
                #we only have one reward, so just return it
                return rewards[0]

   
    #generates probabilities for each action
    #based on modeled advantages
    def regretMatch(self, player, infoset, actions, depth):
        am = self.advModels[player]
        advs = am.predict(infoset)
        actionNums = [actionMap[a] for a in actions]
        probs = []
        for n in actionNums:
            probs.append(max(0, advs[n]))
        if depth == 0 and self.pid == 0:
            print('predicted advantages', [(action, advs[n]) for action, n in zip(actions, actionNums)], file=sys.stderr)
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
            return np.array(probs)

    #adds sample of current strategy
    def updateProbs(self, player, infoset, actions, probs, iter):
        sm = self.stratModels[player]
        sm.addSample(infoset, zip(actions, probs), iter)
