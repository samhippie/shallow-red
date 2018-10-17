#!/usr/bin/env python3

import asyncio
import copy
import math
import numpy as np
import random
import sys

from game import Game
import moves

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

#generates prob table, which can be used as a policy for playing
#ps must not have a game running
#the start states and request types are used to set the game state
#we will try to get to the proper state
#(this will change later when we play past team preview)
#has 2 prob tables as we have 2 separate agents

#returns None if it failed to achieve the start state
#otherwise returns two prob tables
async def mcSearch(ps, teams, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[],
        mcImpl=mcDUCTImpl,
        mcData=[{},{}]):
        #probTable1=collections.defaultdict(lambda: (0,0)),
        #probTable2=collections.defaultdict(lambda: (0,0))):
    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, seed=seed, verbose=False)
        await game.startGame()
        await asyncio.gather(
                mcImpl(game.p1Queue, game.cmdQueue,
                    ">p1", mcData=mcData[0], errorPunishment=2*limit,
                    initActions=p1InitActions),
                mcImpl(game.p2Queue, game.cmdQueue,
                    ">p2", mcData=mcData[1], errorPunishment=2*limit,
                    initActions=p2InitActions))
    print(file=sys.stderr)



