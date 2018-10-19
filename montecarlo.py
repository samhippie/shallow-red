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
#errorPunishment is how many losses an error counts as
#initActions is a list of initial actions that will be blindy taken
#mcData has countTable, which maps (state, action) to count
#mcData has expValueTable, which maps (stat, action) to an expected value
#both should be defaultdict to 0
#mcData has gamma, which is a number [0,1], prob of picking random move
async def mcExp3Impl(requestQueue, cmdQueue, cmdHeader, mcData,
        format, errorPunishment=100, initActions=[]):

    countTable = mcData['countTable']
    expValueTable = mcData['expValueTable']
    gamma = mcData['gamma']
    seenStates = mcData['seenStates']

    #need to track these so we can correct errors
    prevState = None
    prevRequestType = None
    prevAction = None

    #history so we can update probTable
    history = []

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
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
                expValueTable[key] = -1 * errorPunishment
                #scrub the last action from history
                history = history[0:-1]
            else:
                state = request[1]

            seenStates[state] = True
            actions = moves.getMoves(format, state[1])

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

            prevAction = bestAction
            prevState = state
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
            mcData[i]['gamma'] = 0.3
        if 'seenStates' not in mcData[i]:
            mcData[i]['seenStates'] = {}

    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, format=format, seed=seed, verbose=False)
        await game.startGame()
        await asyncio.gather(
                mcExp3Impl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0], format=format, errorPunishment=2*limit,
                    initActions=p1InitActions),
                mcExp3Impl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1], format=format, errorPunishment=2*limit,
                    initActions=p2InitActions))
    print(file=sys.stderr)

#returns the final probabilities of each action in the state
def getProbsExp3(mcData, state, actions):
    countTable = mcData['countTable']
    counts = [countTable[(state, action)] for action in actions]
    expValueTable = mcData['expValueTable']
    totalCount = np.sum(counts)
    probs = np.array([max(0, c - mcData['gamma'] * totalCount / len(actions)) for c in counts])
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
        if expValueTable[(state, action)] >= 0:#negative values are for illegal moves
            xvs.append(xv)
    return np.mean(xvs)


