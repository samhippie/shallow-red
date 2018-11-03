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

#Regret Matching

class RegretMatchAgent:

    def __init__(self, teams, format, valueModel=None, gamma=0.3, initExpVal=0, posReg=False, probScaling=0, regScaling=0, verbose=False):
        self.teams = teams
        self.format = format
        self.initExpVal = initExpVal
        self.posReg = posReg
        self.probScaling = probScaling
        self.regScaling = regScaling
        self.verbose = verbose

        self.valueModel = valueModel if valueModel else model.BasicModel()

        self.mcData = []
        for i in range(2):
            data = {
                'gamma': gamma,
                'regretTable': collections.defaultdict(int),
                'probTable': collections.defaultdict(int),
                'seenStates': {},
                'addReward': self.valueModel.addReward,
                'getExpValue': self.valueModel.getExpValue,
            }
            self.mcData.append(data)

    async def search(self, ps, pid=0, limit=100, seed=None, initActions=[[],[]]):
        await mcSearchRM(
                ps,
                self.format,
                self.teams,
                self.mcData,
                limit=limit,
                seed=seed,
                p1InitActions=initActions[0],
                p2InitActions=initActions[1],
                pid=pid,
                initExpVal=self.initExpVal,
                posReg=self.posReg,
                probScaling=self.probScaling,
                regScaling=self.regScaling,
                verbose=self.verbose)

    def getProbs(self, player, state, actions):
        probTable = self.mcData[player]['probTable']
        probs = np.array([probTable[(state, action)] for action in actions])
        return probs / np.sum(probs)

    def combine(self):
        self.mcData = combineRMData([self.mcData], self.valueModel)[0]


#RM iteration
async def mcRMImpl(requestQueue, cmdQueue, cmdHeader, mcData, otherMcData, format, iter=0, initActions=[], pid=0, initExpVal=0, posReg=True, probScaling=0, regScaling=0, verbose=False):

    regretTable = mcData['regretTable']
    probTable = mcData['probTable']
    gamma = mcData['gamma']
    seenStates = mcData['seenStates']

    #these can be managed somewhere else
    getExpValue = mcData['getExpValue']
    addReward = mcData['addReward']

    #need to do it this way so
    #the other player has access to our history
    #need to keep histories for different processes separate
    mcData['history' + str(pid)] = []
    history = mcData['history' + str(pid)]

    #whether we're maximizing or minimizing
    rewardType = 1 if cmdHeader == '>p1' else 2

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
    inInitActions = True
    while running:
        request = await requestQueue.get()
        if verbose:
            print(cmdHeader, request)

        if request[0] == Game.REQUEST:
            req = request[1]
            state = req['stateHash']
            stateObj = req['state']

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
                exploreProbs = probs * (1-gamma) + gamma / len(actions)
            else:
                #everything is bad, be random
                probs = [1 / len(actions)] * len(actions)
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
            history.append((state, stateObj, bestActionIndex, actions, probs))

            if verbose:
                print('picked', cmdHeader + bestAction)

            #send out the action
            await cmdQueue.put(cmdHeader + bestAction)

        elif request[0] == Game.END:
            #need to use the other player's actions
            otherHistory = otherMcData['history' + str(pid)]

            #update probTable with our history + result
            reward = request[1]
            #rescale reward from [-1,1] to [0,1]
            reward = (reward + 1) / 2
            for i in range(len(history)):
                #get our info
                state, stateObj, actionIndex, actions, probs = history[i]
                action = actions[actionIndex]
                #need the other player's action
                _, _, otherActionIndex, otherActions, _ = otherHistory[i]
                otherAction = otherActions[otherActionIndex]

                for j in range(len(actions)):
                    #update each action's regret
                    #selected action doesn't have its regret changed
                    if j != actionIndex:
                        regret = regretTable[(state, actions[j])]
                        if regScaling != 0:
                            regret *= ((iter+1)**regScaling) / ((iter+1)**regScaling + 1)
                        #need to normalize the action order
                        if rewardType == 1:
                            b1, b2 = actions[j], otherAction
                        else:
                            b1, b2 = otherAction, actions[j]
                        #get the expected value of the action
                        expValue = getExpValue(state, stateObj, b1, b2)
                        if expValue != None and rewardType == 2:
                            expValue = 1 - expValue
                        if expValue == None:
                            expValue = initExpVal
                        #network sometimes spits out bad values
                        expValue = min(1, max(0, expValue))
                        #use expValue to add regret
                        if posReg:
                            regretTable[(state, actions[j])] = max(regret + expValue - reward, 0)
                        else:
                            regretTable[(state, actions[j])] = regret + expValue - reward

                    #update each action's probability
                    probScale = ((iter+1) / (iter + 2))**probScaling
                    oldProb = probTable[(state, actions[j])]
                    probTable[(state, actions[j])] = probScale * oldProb + probs[j]

                #only player 1 updates the rewards
                #we don't actually use the expected value of the
                #chosen set of actions, so it doesn't matter
                #when exactly we update
                if rewardType == 1:
                    addReward(state, stateObj, action, otherAction, reward)

            running = False


#RM loop
#initExpVal is the initial expected value. 0 and 0.5 both make sense
#posReg is to enable only having 0 or positive regret
async def mcSearchRM(ps, format, teams, mcData, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[], pid=0,
        initExpVal=0, posReg=True, probScaling=0, regScaling=0, verbose=False):

    if p1InitActions:
        print(p1InitActions, p2InitActions)
    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=verbose)
        await game.startGame()
        await asyncio.gather(
                mcRMImpl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0],
                    otherMcData = mcData[1], format=format, iter=i,
                    initActions=p1InitActions, pid=pid, initExpVal=initExpVal, posReg=posReg, probScaling=probScaling, regScaling=regScaling,
                    verbose=verbose),
                mcRMImpl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1],
                    otherMcData=mcData[0], format=format, iter=i,
                    initActions=p2InitActions, pid=pid, initExpVal=initExpVal, posReg=posReg, probScaling=probScaling, regScaling=regScaling,
                    verbose=verbose))
    print(file=sys.stderr)

def combineRMData(mcDatasets, valueModel=None):
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


    #combine data on states that were seen in any search
    #in the last iteration
    combMcData = [{
        'rewardTable': collections.defaultdict(int),
        'countTable': collections.defaultdict(int),
        'probTable': collections.defaultdict(int),
        'regretTable': collections.defaultdict(int),
        'seenStates': {},
        'avgGamma': mcDatasets[0][j]['avgGamma'],
        'gamma': mcDatasets[0][j]['gamma']} for j in range(2)]

    for data in mcDatasets:
        #add up each agent's probability and regret
        for j in range(2):
            probTable = data[j]['probTable']
            regretTable = data[j]['regretTable']
            for state, action in probTable:
                if state in seenStates:
                    combMcData[j]['probTable'][(state, action)] += probTable[(state, action)]
                    combMcData[j]['regretTable'][(state, action)] += regretTable[(state, action)]

        #add up the shared counts and rewards

        #these are shared by both agents
        countTable = data[0]['countTable']
        rewardTable = data[0]['rewardTable']

        for state, a1, a2 in countTable:
            if state in seenStates:
                combMcData[0]['countTable'][(state, a1, a2)] += countTable[(state, a1, a2)]
                combMcData[0]['rewardTable'][(state, a1, a2)] += rewardTable[(state, a1, a2)]
        combMcData[1]['countTable'] = combMcData[0]['countTable']
        combMcData[1]['rewardTable'] = combMcData[0]['rewardTable']

    #copy the combined data back into the datasets
    return [copy.deepcopy(combMcData) for j in range(num)]
