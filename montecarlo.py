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

#Decoupled UCT
#this is out of date, probably won't be updated

#runs through a monte carlo playout for a single player
#so two of these need to be running for a 2 player game
#probTable maps (state, action) to (win, count)
#uct_c is the c constant used in the UCT calculation
#errorPunishment is how many losses an error counts as
#initActions is a list of initial actions that will be blindy taken
async def mcDUCTImpl(requestQueue, cmdQueue, cmdHeader, mcData,
        uct_c=1.414, errorPunishment=100, initActions=[]):

    probTable = mcData['probTable']

    #need to track these so we can correct errors
    prevState = None
    prevRequestType = None
    prevAction = None

    #history so we can update probTable
    history = []

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
    randomPlayout = False
    inInitActions = True
    while running:
        request = await requestQueue.get()

        if request[0] == Game.REQUEST or request[0] == Game.ERROR:
            if request[0] == Game.ERROR:
                if len(initActions) > 0:
                    print('WARNING got an error following initActions', file=sys.stderr)
                state = prevState
                #punish the action that led to an error with a bunch of losses
                key = (prevState, prevAction)
                win, count = probTable[key]
                #probTable[key] = win, (count + errorPunishment)
                #let's try giving a bunch of negative wins instead of losses
                #so we don't mess with our total count in the UCT algorithm
                probTable[key] = (win - errorPunishment), (count + 1)
                #scrub the last action from history
                history = history[0:-1]
            else:
                state = request[1]

            if state[1] == Game.REQUEST_TEAM:
                actions = moves.teamSet
            elif state[1] == Game.REQUEST_TURN:
                actions = moves.moveSet# + switchSet

            #check if we ran out of initActions on the previous turn
            #if so, we need to change the PRNG
            if inInitActions and len(initActions) == 0:
                inInitActions = False
                #no problem if both players reset the PRNG
                await cmdQueue.put('>resetPRNG')

            if len(initActions) > 0:
                bestAction = initActions[0]
                initActions = initActions[1:]
            elif randomPlayout:
                bestAction = random.choice(actions)
            else:
                #use upper confidence bound to pick the action
                #see the wikipedia MCTS page for details
                uctVals = []
                total = 0
                bestAction = None

                #need to get the total visit count first
                for action in actions:
                    key = (state, action)
                    win, count = probTable[key]
                    total += count

                #now we find the best UCT
                bestUct = None
                for action in actions:
                    key = (state, action)
                    win, count = probTable[key]
                    #never visited -> infinite UCT
                    #also means we start the random playout
                    if count == 0:
                        bestAction = action
                        randomPlayout = True
                        break
                    uct = win / count + uct_c * math.sqrt(math.log(total) / count)
                    if bestUct == None or uct > bestUct:
                        bestUct = uct
                        bestAction = action

            #save our action
            history.append((state, bestAction))

            prevAction = bestAction
            prevState = state
            #send out the action
            await cmdQueue.put(cmdHeader + bestAction)

        elif request[0] == Game.END:
            #update probTable with our history + result
            reward = request[1]
            for key in history:
                win, count = probTable[key]
                if reward == -1:
                    probTable[key] = (win, count+1)
                elif reward == 1:
                    probTable[key] = (win+1, count+1)
                elif reward == 0: # we shouldn't be seeing any ties in the MCTS loop, as I don't think you can get ties without a timer
                    print('WARNING tie in MCTS', file=sys.stderr)
                    probTable[key] = (win+0.5, count+1)

            running = False




#Decoupled UCT
#generates prob table, which can be used as a policy for playing
#ps must not have a game running
#the start states and request types are used to set the game state
#we will try to get to the proper state
#(this will change later when we play past team preview)
#has 2 prob tables as we have 2 separate agents

#returns None if it failed to achieve the start state
#otherwise returns two prob tables
async def mcSearchDUCT(ps, teams, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[],
        mcData=[{},{}]):
    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, seed=seed, verbose=False)
        await game.startGame()
        await asyncio.gather(
                mcDUCTImpl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0], errorPunishment=2*limit,
                    initActions=p1InitActions),
                mcImpl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1], errorPunishment=2*limit,
                    initActions=p2InitActions))
    print(file=sys.stderr)


#Exp3
#initActions is a list of initial actions that will be blindy taken
#mcData has countTable, which maps (state, action) to count
#mcData has expValueTable, which maps (stat, action) to an expected value
#both should be defaultdict to 0
#mcData has gamma, which is a number [0,1], prob of picking random move
#iter is the iteration number, which may be used to compute gamma
async def mcExp3Impl(requestQueue, cmdQueue, cmdHeader, mcData,
        format, iter=0, initActions=[]):

    countTable = mcData['countTable']
    expValueTable = mcData['expValueTable']
    gamma = mcData['gamma'](iter)
    seenStates = mcData['seenStates']

    #history so we can update probTable
    history = []

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
    inInitActions = True
    while running:
        request = await requestQueue.get()

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

            #prevAction = bestAction
            #prevState = state
            #send out the action
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
        seed=None, p1InitActions=[], p2InitActions=[]):
    for i in range(len(mcData)):
        if 'countTable' not in mcData[i]:
            mcData[i]['countTable'] = collections.defaultdict(int)
        if 'expValueTable' not in mcData[i]:
            mcData[i]['expValueTable'] = collections.defaultdict(int)
        if 'gamma' not in mcData[i]:
            mcData[i]['gamma'] = lambda x: 0.3
        if 'seenStates' not in mcData[i]:
            mcData[i]['seenStates'] = {}

    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=False)
        await game.startGame()
        await asyncio.gather(
                mcExp3Impl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0], format=format,
                    initActions=p1InitActions),
                mcExp3Impl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1], format=format,
                    initActions=p2InitActions))
    print(file=sys.stderr)

def combineExp3Data(mcDatasets):
    num = len(mcDatasets)
    #record which states were seen in the last iteration
    seenStates = {}
    for data in mcDatasets:
        for j in range(2):
            seen = data[j]['seenStates']
            for state in seen:
                seenStates[state] = True

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

    #if average gamma is specified, use it
    #otherwise assume gamma is constant
    if 'avgGamma' in mcData:
        avgGamma = mcData['avgGamma']
    else:
        avgGamma = mcData['gamma'](0)
    probs = np.array([max(0, c - avgGamma * totalCount / len(actions)) for c in counts])
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



#Regret Matching


#RM iteration
async def mcRMImpl(requestQueue, cmdQueue, cmdHeader, mcData, otherMcData, format, iter=0, initActions=[], pid=0, initExpVal=0, posReg=True, probScaling=0, regScaling=0):

    regretTable = mcData['regretTable']
    rewardTable = mcData['rewardTable']
    rewardType = mcData['rewardType']
    countTable = mcData['countTable']
    probTable = mcData['probTable']
    gamma = mcData['gamma'](iter)
    seenStates = mcData['seenStates']

    #these can be managed somewhere else
    getExpValue = mcData['getExpValue']
    addReward = mcData['addReward']

    #need to do it this way so
    #the other player has access to our history
    #need to keep histories for different processes separate
    mcData['history' + str(pid)] = []
    history = mcData['history' + str(pid)]

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
    inInitActions = True
    while running:
        request = await requestQueue.get()

        if request[0] == Game.REQUEST or request[0] == Game.ERROR:
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
                #the paper
                #http://mlanctot.info/files/papers/cig14-smmctsggp.pdf
                #says r / rSum, but I think it's
                #supposed to by max(0,r) / rSum
                #as in these slide
                #https://ocw.mit.edu/courses/sloan-school-of-management/15-s50-poker-theory-and-analytics-january-iap-2015/lecture-notes/MIT15_S50IAP15_L7_GameTheor.pdf
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
                preAction = initActions[0]
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

            #prevAction = bestAction
            #prevState = state
            #send out the action
            await cmdQueue.put(cmdHeader + bestAction)

        elif request[0] == Game.END:
            #need to use the other player's actions
            otherHistory = otherMcData['history' + str(pid)]

            """
            expValueCache = {}
            inputSet = []
            #go ahead and get all the expValue inputs that we need
            for i in range(len(history)):
                state, stateObj, actionIndex, actions, probs = history[i]
                action = actions[actionIndex]
                _, _, otherActionIndex, otherActions, _ = otherHistory[i]
                otherAction = otherActions[otherActionIndex]

                for j in range(len(actions)):
                    if rewardType == 1:
                        b1, b2 = actions[j], otherAction
                    else:
                        b1, b2 = otherAction, actions[j]
                    inputSet.append((state, stateObj, b1, b2))
            #calculate all the expValues at once
            expValueSet = getExpValue(bulk_input=inputSet)
            for i in range(len(inputSet)):
                state, _, b1, b2 = inputSet[i]
                expValueCache[(state, b1, b2)] = expValueSet[i][0]
            """

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
                        #if (state, b1, b2) in expValueCache:
                            #expValue = expValueCache[(state, b1, b2)]
                        #else:
                        expValue = getExpValue(state, stateObj, b1, b2)
                        #count = countTable[(state, b1, b2)]
                        #cumReward = rewardTable[(state, b1, b2)]
                        if expValue != None and rewardType == 2:
                            #cumReward = count - cumReward
                            expValue = 1 - expValue
                        if expValue == None:
                            expValue = initExpVal
                        #network sometimes spits out bad values
                        expValue = max(0, expValue)
                        #expValue = initExpVal if count == 0 else cumReward / count
                        #use expValue to add regret
                        if posReg:
                            regretTable[(state, actions[j])] = max(regret + expValue - reward, 0)
                        else:
                            regretTable[(state, actions[j])] = regret + expValue - reward

                    #update each action's probability
                    probScale = ((iter+1) / ((iter + 2)))**probScaling
                    oldProb = probTable[(state, actions[j])]
                    probTable[(state, actions[j])] = probScale * oldProb + probs[j]
                #probTable[(state, actions[actionIndex])] += probs[actionIndex]

                #only player 1 updates the rewards
                #we don't actually use the expected value of the
                #chosen set of actions, so it doesn't matter
                #when exactly we update
                if rewardType == 1:
                    addReward(state, stateObj, action, otherAction, reward)
                    #countTable[(state, action, otherAction)] += 1
                    #rewardTable[(state, action, otherAction)] += reward
                    #if 'valueRecord' in mcData:
                        #mcData['valueRecord'].append(((stateObj, action, otherAction), reward))

            running = False


#RM loop
#initExpVal is the initial expected value. 0 and 0.5 both make sense
#posReg is to enable only having 0 or positive regret
async def mcSearchRM(ps, format, teams, mcData, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[], pid=0, initExpVal=0, posReg=True, probScaling=0, regScaling=0):
    #these are shared for both players
    #reward is for player 1, so player 2 should use 1-r
    rewardTable = collections.defaultdict(int)
    countTable = collections.defaultdict(int)
    for i in range(len(mcData)):
        if 'regretTable' not in mcData[i]:
            mcData[i]['regretTable'] = collections.defaultdict(int)

        if 'rewardTable' not in mcData[i]:
            mcData[i]['rewardTable'] = rewardTable
        if 'rewardType' not in mcData[i]:
            if i == 0:
                mcData[i]['rewardType'] = 1
            else:
                mcData[i]['rewardType'] = 2
        if 'countTable' not in mcData[i]:
            mcData[i]['countTable'] = countTable

        if 'probTable' not in mcData[i]:
            mcData[i]['probTable'] = collections.defaultdict(int)

        if 'gamma' not in mcData[i]:
            mcData[i]['gamma'] = lambda x: 0.3
        if 'seenStates' not in mcData[i]:
            mcData[i]['seenStates'] = {}

    #make sure counts and rewards are shared
    mcData[1]['countTable'] = mcData[0]['countTable']
    mcData[1]['rewardTable'] = mcData[0]['rewardTable']

    #((state, action, action), reward) tuples
    if 'valueRecord' not in mcData[0]:
        mcData[0]['valueRecord'] = []

    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=False)
        await game.startGame()
        await asyncio.gather(
                mcRMImpl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0],
                    otherMcData = mcData[1], format=format, iter=i,
                    initActions=p1InitActions, pid=pid, initExpVal=initExpVal, posReg=posReg, probScaling=probScaling, regScaling=regScaling),
                mcRMImpl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1],
                    otherMcData=mcData[0], format=format, iter=i,
                    initActions=p2InitActions, pid=pid, initExpVal=initExpVal, posReg=posReg, probScaling=probScaling, regScaling=regScaling))
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
            for state, action in probTable:
                if state not in seenStates:
                    del probTable[(state, action)]
                    del regretTable[(state, action)]

        countTable = data[0]['countTable']
        rewardTable = data[0]['rewardTable']

        for state, a1, a2 in countTable:
            if state not in seenStates:
                del countTable[(state, a1, a2)]
                del rewardTable[(state, a1, a2)]
        data[1]['countTable'] = countTable
        data[1]['rewardTable'] = rewardTable
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

def getProbsRM(mcData, state, actions):
    probTable = mcData['probTable']
    probs = np.array([probTable[(state, action)] for action in actions])
    return probs / np.sum(probs)

#doing an expected value for RM would be too difficult
#as it would require both players' moves


