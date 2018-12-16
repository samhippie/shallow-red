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

from deep.deepModel import DeepCfrModel
import deep.dataStorage

#Deep MCCFR

#based on this paper
#https://arxiv.org/pdf/1811.00164.pdf
#not sure if this will be viable

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
            teams,
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

        self. teams = teams
        self.format = format

        self.pid = -1

        self.resumeIter = resumeIter

        self.writeLock = writeLock
        self.trainingBarrier = trainingBarrier
        self.sharedDict = sharedDict

        if resumeIter == None:
            #fresh start, delete old data
            deep.dataStorage.clearData()
            

        if advModels:
            self.advModels = advModels
        else:
            self.advModels = [DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]

        if stratModels:
            self.stratModels = stratModels
        else:
            self.stratModels = [DeepCfrModel(name='strat' + str(i), softmax=True, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]

        self.advEpochs = advEpochs
        self.stratEpochs = stratEpochs

        self.branchingLimit = branchingLimit
        self.depthLimit = depthLimit

        self.verbose = verbose

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[], []]):
        self.pid = pid
        #turn init actions into a useful history
        history = [(None, a1, a2) for a1, a2 in zip(*initActions)]
        #insert the seed in the first turn
        if len(history) > 0:
            _, a1, a2 = history[0]
            history[0] = (seed, a1, a2)

        start = self.resumeIter if self.resumeIter else 0

        if self.pid == 0:
            print(end='', file=sys.stderr)
        for i in range(start, limit):
            if self.pid == 0:
                print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)

            #for self.small games, this is necessary to get a decent number of samples
            for j in range(1):
                #need idMap to be the same across processes
                if self.pid == 0:
                    self.sharedDict['idMap'] = modelInput.idMap
                elif 'idMap' in self.sharedDict:
                    modelInput.idMap = self.sharedDict['idMap']

                game = Game(ps, self.teams, format=self.format, seed=seed, verbose=self.verbose)
                await game.startGame()
                await game.applyHistory(history)
                await self.cfrRecur(ps, game, seed, history, i)

            #save our adv data after each iteration
            #so the non-zero pid workers don't have data cached
            self.advModels[i % 2].clearSampleCache()

            #only need to train about once per iteration
            #and as long as pid 0 doesn't finish too early this will be fine
            self.trainingBarrier.wait()
            if self.pid == 0:
                self.advTrain(i % 2)
                self.sharedDict['advNet' + str(i % 2)] = self.advModels[i % 2].net
            self.trainingBarrier.wait()
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

    #closes the models, which have data to clean up
    def close(self):
        for model in self.advModels + self.stratModels:
            model.close()

    #recursive implementation of cfr
    #history is a list of (seed, action, action) tuples
    #q is the sample probability
    #assumes the game has already had the history applied
    async def cfrRecur(self, ps, game, startSeed, history, iter, depth=0, rollout=False):
        async def endGame():
            side = 'bot1' if iter % 2 == 0 else 'bot2'
            winner = await game.winner
            #have to clear the results out of the queues
            while not game.p1Queue.empty():
                await game.p1Queue.get()
            while not game.p2Queue.empty():
                await game.p2Queue.get()
            #the deep cfr paper uses [-1,1] rather than [0,1] for u
            #but I like [0,1]
            if winner == side:
                return 1
            else:
                return -1

        if depth >= self.depthLimit:
            rollout = True

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
        probs = self.regretMatch(offPlayer, state, actions, -1)
        if depth == 0 and self.pid == 0:
            print('player ' + str(offPlayer) + ' probs', list(zip(actions, probs)), file=sys.stderr)
        offAction = np.random.choice(actions, p=probs)
        #and update average stategy
        #we should be okay adding this for rollouts
        #but I'm testing skipping rollouts
        self.updateProbs(offPlayer, state, actions, probs, iter // 2 + 1)

        #on player
        request = (await queues[onPlayer].get())
        if request[0] == Game.END:
            return await endGame()
        req = request[1]

        state = req['state']
        actions = moves.getMoves(self.format, req)
        probs = self.regretMatch(onPlayer, state, actions, depth)
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

            r = await self.cfrRecur(ps, game, startSeed, history + [historyEntry], iter, depth=depth+1, rollout=curRollout)
            rewards.append(r)

        if not rollout:
            #save sample of advantages
            stateExpValue = 0
            for p,r in zip(probs, rewards):
                stateExpValue += p * r
            advantages = [r - stateExpValue for r in rewards]

            am = self.advModels[onPlayer]
            am.addSample(state, zip(actions, advantages), iter // 2 + 1)

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
    def regretMatch(self, player, state, actions, depth):
        am = self.advModels[player]
        advs = am.predict(state)
        actionNums = [modelInput.enumAction(a) for a in actions]
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
    def updateProbs(self, player, state, actions, probs, iter):
        sm = self.stratModels[player]
        sm.addSample(state, zip(actions, probs), iter)
