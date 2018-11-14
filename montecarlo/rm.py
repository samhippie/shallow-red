#!/usr/bin/env python3

import asyncio
import collections
import copy
import math
import numpy as np
import random
from sqlitedict import SqliteDict
import sys

from game import Game
import model
import moves

#MC Regret Matching (technically MCCFR with Outcome Samplingling, I think)

#options for how to store the table
#memory is faster, but db allows for more data
MEMORY = 1
DB = 2

class RegretMatchAgent:

    #exploration is the probability of taking a random action instead of
    #following the regret matching distribution

    #initExpVal is the expected value for state-action-actions that haven't
    #been seen before
    #the value applies to both players instead of being zero sum

    #DCFR parameters:
    #posReg is whether when floor cumulative regret at 0 or not
    #probScaling determines how much to weight to give to later iterations'
    #contributions to the final strategy's probability
    #regScaling is like probScaling but for contributions to cumulative regret

    #tableType determines how we store our data

    #dbLocation is the sqlite db file
    #dbClear is whether to clear the db before running
    #this should be done if any parameters change, or if we just want a fresh
    #start
    def __init__(self, teams, format,
            exploration=0.3, initExpVal=0,
            posReg=False, probScaling=0, regScaling=0,
            tableType=MEMORY,
            dbLocation='./rm-agent', dbClear=True,
            verbose=False):

        self.teams = teams
        self.format = format

        self.exploration = exploration
        self.initExpVal = initExpVal

        #DCFR parameters
        self.posReg = posReg
        self.probScaling = probScaling
        self.regScaling = regScaling

        self.verbose = verbose

        self.tableType = tableType

        #I'm hardcoding reward and count as tables
        #There hasn't been enough success with models to justify
        #abstracting them out

        if tableType == MEMORY:
            self.regretTables = [{}, {}]
            self.probTables = [{}, {}]
            self.rewardTable = {}
            self.countTable = {}
        elif tableType == DB:
            #I was planning on leaving autocommit off, but there are some bugs
            autocommit = True

            #cumulative regret, (state, action) -> regret
            self.regretTables = []
            #cumulative probability, (state, action) -> (non-normalized) probability
            self.probTables = []
            for i in range(2):
                self.regretTables.append(
                    SqliteDict(dbLocation, tablename='regret' + str(i), autocommit=autocommit))
                self.probTables.append(
                    SqliteDict(dbLocation, tablename='prob' + str(i), autocommit=autocommit))
                if dbClear:
                    self.regretTables[i].clear()
                    self.probTables[i].clear()

            #cumulative reward, (state, action1, action2) -> p1's cumulative reward
            self.rewardTable = SqliteDict(dbLocation, tablename='reward', autocommit=autocommit)
            #count, (state, action1, action2) -> count
            self.countTable = SqliteDict(dbLocation, tablename='count', autocommit=autocommit)

            if dbClear:
                self.rewardTable.clear()
                self.countTable.clear()

    def close(self):
        if self.tableType == DB:
            for i in range(2):
                self.regretTables[i].close()
                self.probTables[i].close()
            self.rewardTable.close()
            self.countTable.close()

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[], []]):

        #this throws away some good data, but I'm not sure it will matter too much
        #right now memory usage is our biggest enemy
        if self.tableType == MEMORY:
            self.regretTables = [{}, {}]
            self.probTables = [{}, {}]
            self.rewardTable = {}
            self.countTable = {}


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
            await self.rmRecur(ps, game, seed, i)
        print(file=sys.stderr)

    def combine(self):
        #no need to combine when we're using a DB
        pass

    #for getting final action probabilites
    def getProbs(self, player, state, actions):
        pt = self.probTables[player]
        probs = np.array([dictGet(pt, (state, a)) for a in actions])
        pSum = np.sum(probs)
        if pSum > 0:
            return probs / np.sum(probs)
        else:
            return np.array([1 / len(actions) for a in actions])

    #recursive implementation of rm
    #assumes the game is running and is caught up
    #returns p1's expected reward for the playthrough
    #(which is just the actual reward for rm)
    async def rmRecur(self, ps, game, startSeed, iter, depth=0):

        cmdHeaders = ['>p1', '>p2']
        queues = [game.p1Queue, game.p2Queue]
        #all the actions both players can pick
        playerActions = [[], []]
        #the probabilitiy of picking each action
        playerProbs = [[], []]
        #the indices of the actions actually picked
        pickedActions = [0, 0]
        #both players make their move
        for i in range(2):
            request = (await queues[i].get())
            if request[0] == Game.END:
                winner = await game.winner
                #have to clear the results out of the queues
                while not game.p1Queue.empty():
                    await game.p1Queue.get()
                while not game.p2Queue.empty():
                    await game.p2Queue.get()
                if winner == 'bot1':
                    return 1
                else:
                    return 0

            req = request[1]
            state = req['stateHash']
            #get rm probs for the actions
            actions = moves.getMoves(self.format, req)
            playerActions[i] = actions
            probs = self.regretMatch(i, state, actions)
            #add exploration, which adds a chance to play randomly
            exploreProbs = probs * (1 - self.exploration) + self.exploration / len(probs)
            playerProbs[i] = probs
            #and sample one action
            pickedActions[i] = np.random.choice(len(actions), p=exploreProbs)


        #apply the picked actions to the game
        seed = Game.getSeed()
        await game.cmdQueue.put('>resetPRNG ' + str(seed))
        for i in range(2):
            pickedAction = pickedActions[i]
            action = playerActions[i][pickedAction]
            await game.cmdQueue.put(cmdHeaders[i] + action)

        #get the reward so we can update our regrets
        reward = await self.rmRecur(ps, game, startSeed, iter, depth=depth+1)

        #save the reward
        a1 = playerActions[0][pickedActions[0]]
        a2 = playerActions[1][pickedActions[1]]
        self.addReward(state, a1, a2, reward)

        #need to update both players' regret and strategy
        for i in range(2):
            #update each action's regret and probability in average strategy
            rt = self.regretTables[i]
            pt = self.probTables[i]
            actions = playerActions[i]
            for j in range(len(actions)):
                #update stategy with this iteration's strategy
                #which just means adding the current probability of each action
                probScale = ((iter+1) / (iter+2))**self.probScaling
                prob = dictGet(pt, (state, actions[j]))
                pt[hash((state, actions[j]))] = probScale * prob + playerProbs[i][j]

                #immediate regret of picked actions is 0, so just skip
                if j == pickedActions[i]:
                    continue

                #get existing regret so we can add to it
                regret = dictGet(rt, (state, actions[j]))
                if self.regScaling != 0:
                    regret *= ((iter+1)**self.regScaling) / ((iter+1)**self.regScaling + 1)
                #get i's possible action and -i's actual action
                #in player order
                if i == 0:
                    a1 = actions[j]
                    a2 = playerActions[1][pickedActions[1]]
                    myReward = reward
                else:
                    a1 = playerActions[0][pickedActions[0]]
                    a2 = actions[j]
                    myReward = 1 - reward

                #get expected value for the potential turn
                expValue = self.getExpValue(i, state, a1, a2)

                #add immediate regret
                if self.posReg:
                    rt[hash((state, actions[j]))] = max(regret + expValue - myReward, 0)
                else:
                    rt[hash((state, actions[j]))] = regret + expValue - myReward

        #pass the actual reward up
        return reward

    #generates probabilities for each action
    def regretMatch(self, player, state, actions):
        rt = self.regretTables[player]
        regrets = np.array([max(0, dictGet(rt, (state, a))) for a in actions])
        rSum = np.sum(regrets)
        if rSum > 0:
            return regrets / rSum
        else:
            return np.array([1 / len(actions) for a in actions])

    #saves the actual reward so it can be used by getExpValue later
    def addReward(self, state, a1, a2, reward):
        rewardSum = dictGet(self.rewardTable, (state, a1, a2))
        count = dictGet(self.countTable, (state, a1, a2))

        self.rewardTable[hash((state, a1, a2))] = rewardSum + reward
        self.countTable[hash((state, a1, a2))] = count + 1

    #returns expected reward for the (state, action, action) tuple
    #which is just the tabular value
    def getExpValue(self, player, state, a1, a2):
        reward = dictGet(self.rewardTable, (state, a1, a2))
        count = dictGet(self.countTable, (state, a1, a2))
        if count == 0:
            return self.initExpVal
        else:
            expValue = reward / count
            #player 2 is the minimizing player
            if player == 1:
                expValue = 1 - expValue
            return expValue

#convenience method, treats dict like defaultdict(int)
#which is needed for sqlitedict
#there's probably a better way
def dictGet(table, key):
    #sqlite is stricter about keys, so we have to use a hash
    key = hash(key)
    if not key in table:
        table[key] = 0
    return table[key]
