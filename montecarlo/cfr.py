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

#MCCFR with either External Sampling or Average Sampling

#sampling types
EXTERNAL = 1
AVERAGE = 2

#early state evaluation types
HEURISTIC = 1
ROLLOUT = 2
MODEL = 3

class CfrAgent:
    #AS parameters:
    #exploration gives all action a chance to be taken
    #bonus is for early exploration
    #threshold is so any action with prob > 1/threshold is always taken
    #bound is the maximum number of actions that can be taken, 0 for disabled

    #depth limit (if not None) replaces tree traversal with evaluation option
    #evaluation: HEURISTIC is expValueHeurisitic(), rollout does a rollout, model uses an evalModel (to be implemented)
    def __init__(self, teams, format,
            samplingType=EXTERNAL, exploration=0, bonus=0, threshold=1, bound=0,
            posReg=False, probScaling=0, regScaling=0,
            depthLimit=None, evaluation=HEURISTIC, evalModel=None,
            verbose=False):
        self. teams = teams
        self.format = format

        self.samplingType = samplingType
        self.exploration = exploration
        self.bonus = bonus
        self.threshold = threshold
        self.bound = bound

        self.posReg = posReg
        self.probScaling = probScaling
        self.regScaling = regScaling

        self.depthLimit = depthLimit
        self.evaluation = evaluation
        self.evalModel = evalModel

        self.verbose = verbose

        self.seenStates = {}

        self.numActionsSeen = 0
        self.numActionsTaken = 0

        self.regretTables = [collections.defaultdict(int) for i in range(2)]
        self.probTables = [collections.defaultdict(int) for i in range(2)]

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
            self.numActionsSeen = 0
            self.numActionsTaken = 0
            await self.cfrRecur(ps, game, seed, history, 1, i)
            #print('num actions seen:', self.numActionsSeen, file=sys.stderr)
            #print('num actions taken:', self.numActionsTaken, file=sys.stderr)
            #print('percentage:', 100 * self.numActionsTaken / max(1, self.numActionsSeen), file=sys.stderr)
            #print(file=sys.stderr)
        print(file=sys.stderr)

    def combine(self):
        #delete probs and regrets for unseen states for both players
        for i in range(2):
            keys = list(self.regretTables[i])
            for state, action in keys:
                if not state in self.seenStates:
                    del self.regretTables[i][(state, action)]
                    if (state, action) in self.probTables[i]:
                        del self.probTables[i][(state, action)]

        self.seenStates = {}

    def getProbs(self, player, state, actions):
        pt = self.probTables[player]
        probs = np.array([pt[(state, a)] for a in actions])
        pSum = np.sum(probs)
        if pSum > 0:
            return probs / np.sum(probs)
        else:
            return np.array([1 / len(actions) for a in actions])

    #recursive implementation of cfr
    #history is a list of (seed, action, action) tuples
    #q is the sample probability
    #assumes the game has already had the history applied
    async def cfrRecur(self, ps, game, startSeed, history, q, iter, depth=0, rollout=False):
        #I'm not sure about this q parameter
        #I'm getting better results setting it to 1 in all games
        q = 1
        async def endGame():
            side = 'bot1' if iter % 2 == 0 else 'bot2'
            winner = await game.winner
            #have to clear the results out of the queues
            while not game.p1Queue.empty():
                await game.p1Queue.get()
            while not game.p2Queue.empty():
                await game.p2Queue.get()
            if winner == side:
                return 1 / q
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
        self.seenStates[state] = True
        actions = moves.getMoves(self.format, req)
        #just sample a move
        probs = self.regretMatch(offPlayer, state, actions)
        offAction = np.random.choice(actions, p=probs)
        #and update average stategy
        self.updateProbs(offPlayer, state, actions, probs / q, iter)

        #on player
        request = (await queues[onPlayer].get())
        if request[0] == Game.END:
            return await endGame()
        req = request[1]

        #now that we've checked if the game is over,
        #let's check depth before continuing
        if self.depthLimit != None and depth >= self.depthLimit:
            if self.evaluation == HEURISTIC:
                await game.cmdQueue.put('>forcewin p1')
                #clean up the end game messages
                await queues[onPlayer].get()
                await queues[offPlayer].get()
                return expValueHeuristic(onPlayer, req['state']) / q
            elif self.evaluation == ROLLOUT:
                rollout = True
                #rest of rollout is implemented with the normal code path
            elif self.evaluation == MODEL:
                #TODO
                pass


        state = req['stateHash']
        actions = moves.getMoves(self.format, req)
        #we sometimes bias towards the first or last actions
        #this fixes that bias
        random.shuffle(actions)
        #probs is the set of sample probabilities, used for traversing
        #iterProbs is the set of probabilities for this iteration's strategy, used for regret
        if rollout:
            #I'm not sure if using regret matching or going uniform random
            #would be better
            #my gut says regret matching
            probs = self.regretMatch(onPlayer, state, actions)
            action = np.random.choice(actions, p=probs)
            actions = [action]
            probs = [1] # would it be better to use the actual probability?
            iterProbs = probs
        elif self.samplingType == EXTERNAL:
            probs = self.regretMatch(onPlayer, state, actions)
            iterProbs = probs
        elif self.samplingType == AVERAGE:
            iterProbs = self.regretMatch(onPlayer, state, actions)
            #let's try using current iterative strategy instead of average strategy
            probs = iterProbs + self.exploration
            """
            stratSum = 0
            strats = []
            pt = self.probTables[onPlayer]
            for a in actions:
                s = pt[(state, a)]
                stratSum += s
                strats.append(s)
            probs = []
            for a,s in zip(actions, strats):
                if self.bonus + stratSum == 0:
                    p = 0
                else:
                    p = (self.bonus + self.threshold * s) / (self.bonus + stratSum)
                p = max(self.exploration, p)
                probs.append(p)
            """
            #keep track of how many actions we take from this state
            numTaken = 0

        #get expected reward for each action
        rewards = []
        gameUsed = False
        self.numActionsSeen += len(actions)
        #whether a specific action is a rollout
        curRollout = rollout
        for action, prob in zip(actions, probs):
            #for ES we just check every action
            #for AS use a roll to determine if we search
            if self.samplingType == AVERAGE and not curRollout:
                #instead of skipping, try making the skipped entries a rollout
                #like in https://www.aaai.org/ocs/index.php/AAAI/AAAI12/paper/viewFile/4937/5469
                #if we're at the last action and we haven't done anything, do something regardless of roll
                if (self.bound != 0 and numTaken > self.bound) or random.random() >= prob and (action != actions[-1] or gameUsed):
                    curRollout = True
                    #rewards.append(0)
                    #continue
                else:
                    curRollout = rollout
                    numTaken += 1
            self.numActionsTaken += 1
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

            r = await self.cfrRecur(ps, game, startSeed, history + [historyEntry], q * min(1, max(0.01, prob)), iter, depth=depth+1, rollout=curRollout)
            rewards.append(r)

        #update regrets
        stateExpValue = 0
        for p,r in zip(iterProbs, rewards):
            stateExpValue += p * r
        rt = self.regretTables[onPlayer]
        for a,r in zip(actions, rewards):
            regret = rt[(state, a)]
            if self.regScaling != 0:
                regret *= ((iter//2 + 1)**self.regScaling) / ((iter//2 + 1)**self.regScaling + 1)
            if self.posReg:
                rt[(state, a)] = max(0, regret + r - stateExpValue)
            else:
                rt[(state, a)] = regret + r - stateExpValue

        return stateExpValue

    #generates probabilities for each action
    def regretMatch(self, player, state, actions):
        rt = self.regretTables[player]
        rSum = 0
        regrets = np.array([max(0, rt[(state, a)]) for a in actions])
        rSum = np.sum(regrets)
        return regrets / rSum if rSum > 0 else np.array([1/len(actions) for a in actions])

    #updates the average strategy for the player
    def updateProbs(self, player, state, actions, probs, iter):
        pt = self.probTables[player]
        probScale = ((iter//2 + 1) / (iter//2 + 2))**self.probScaling
        for a, p in zip(actions, probs):
            oldProb = pt[(state, a)]
            pt[(state, a)] = oldProb * probScale + p

#returns a value in [0,1] that is a heuristic for the expected value
def expValueHeuristic(player, stateObj):
    #ratio of player's hp percentage to combined hp percentage
    #laplace prior ignorance applied (not sure what the proper term is)
    totalHp = 2
    playerHp = 1
    for i in range(2):
        mons = stateObj['players'][i]['mons']
        hpSum = sum([mons[id]['hp'] for id in mons])
        if i == player:
            playerHp += hpSum
        totalHp += hpSum
    return playerHp / totalHp

