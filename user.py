#!/usr/bin/env python3

#used for playing games with a human player

import asyncio
import collections
import copy
import math
import numpy as np
import os
import random
import sys
import subprocess

import moves
from game import Game
import model
import modelInput
import montecarlo as mc

humanTeams = [

    '|ditto|choicescarf|H|transform|Timid|252,,4,,,252||,0,,,,|||]|zygarde|earthplate|H|thousandarrows,coil,extremespeed,bulldoze|Impish|252,252,,,4,|||||]|dragonite|lifeorb|H|extremespeed,outrage,icepunch,dragondance|Jolly|4,252,,,,252|M||||',

    '|gliscor|toxicorb|H|earthquake,toxic,substitute,protect|Jolly|252,156,,,100,|M||||]|tapulele|focussash||moonblast,psychic,calmmind,shadowball|Timid|4,,,252,,252||,0,,,,|||]|gyarados|sitrusberry||waterfall,icefang,stoneedge,dragondance|Adamant|156,252,,,,100|M||||',

]

ovoTeams = [
    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|M||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,|M|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,icebeam|Quiet|252,4,,252,,|M||||',

    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|M||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,|M|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,icebeam|Quiet|252,4,,252,,|M||||',
]


#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

#used for playing games turn-by-turn
#assumes the bot is player 2
class InteractivePlayer:
    def __init__(self, teams, limit=100, numProcesses=1, format='1v1'):
        self.teams = teams
        self.limit = limit
        self.numProcesses = numProcesses
        self.format = format

        def gamma(iter):
            return 0.3

        self.valueModel = model.BasicModel()

        self.mcDataset = [{
            'gamma': gamma,
            'getExpValue': self.valueModel.getExpValue,
            'addReward': self.valueModel.addReward,
        }, {
            'gamma': gamma,
            'getExpValue': self.valueModel.getExpValue,
            'addReward': self.valueModel.addReward,
        }]

        self.probCutoff = 0.03

    #given the request dict, returns a legal action
    #e.g. ' move 1' in singles
    async def getAction(self, request):
        #get info to replicate the current game state
        seed = request['state']['startingSeed']
        initActions = request['state']['actions']

        try:
            searchPs = [await getPSProcess() for i in range(self.numProcesses)]
            searches = []
            for j in range(self.numProcesses):
                search = mc.mcSearchRM(
                        searchPs[j],
                        self.format,
                        self.teams,
                        limit=self.limit,
                        seed=seed,
                        p1InitActions=initActions[0],
                        p2InitActions=initActions[1],
                        mcData=self.mcDataset,
                        pid=j,
                        initExpVal=0,
                        probScaling=2,
                        regScaling=1.5)
                searches.append(search)

            print('searching', file=sys.stderr)
            await asyncio.gather(*searches)

            print('combining', file=sys.stderr)
            self.mcDataset = mc.combineRMData([self.mcDataset], self.valueModel)[0]


            #figure out what kind of action we need
            state = request['stateHash']
            actions = moves.getMoves(format, request)

            data = self.mcDataset[1]
            probs = mc.getProbsRM(data, state, actions)
            #remove low probability moves, likely just noise
            #this can remove every action, but if that's the case then it's doesn't really matter
            #as all the probabilites are low
            normProbs = np.array([p if p > self.probCutoff else 0 for p in probs])
            normProbs = normProbs / np.sum(normProbs)

            action = np.random.choice(actions, p=normProbs)
            return action
        finally:
            for ps in searchPs:
                ps.terminate()



#used to play a game with the user via cli
#assumes RM, so no parallelism
#human is p1, bot is p2
async def humanGame(teams, limit=100, numProcesses=1, format='1v1', valueModel=None, file=sys.stdout, initMoves=([],[])):
    try:

        mainPs = await getPSProcess()

        searchPs = [await getPSProcess() for i in range(numProcesses)]

        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

        game = Game(mainPs, format=format, teams=teams, seed=seed, names=['meat sack', 'Your Robot Overlords'], verbose=True, file=file)



        #holds the montecarlo data
        #each entry goes to one process
        #this will get really big with n=3 after 20ish turns if you
        #don't purge old data
        mcData = []

        def gamma(iter):
            return 0.3

        if not valueModel:
            valueModel = model.BasicModel()

        mcDataset = [{
            'gamma': gamma,
            'getExpValue': valueModel.getExpValue,
            'addReward': valueModel.addReward,
        }, {
            'gamma': gamma,
            'getExpValue': valueModel.getExpValue,
            'addReward': valueModel.addReward,
        }]


        #moves with probabilites below this are not considered
        probCutoff = 0.03

        await game.startGame()

        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            #we reassign this later, so we have to declare it nonlocal
            nonlocal mcDataset
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)

                #don't search if we aren't going to use the results
                if len(initMoves[0]) == 0 or len(initMoves[1]) == 0:
                    #I suspect that averaging two runs together will
                    #give us more improvement than running for twice as long
                    #and it should run faster than a single long search due to parallelism

                    searches = []
                    for j in range(numProcesses):
                        search = mc.mcSearchRM(
                                searchPs[j],
                                format,
                                teams,
                                limit=limit,
                                seed=seed,
                                p1InitActions=p1Actions,
                                p2InitActions=p2Actions,
                                mcData=mcDataset,
                                pid=j,
                                initExpVal=0,
                                probScaling=2,
                                regScaling=1.5)
                        searches.append(search)

                    await asyncio.gather(*searches)

                    #combine the processes results together, purge unused information
                    #this assumes that any state that isn't seen in two consecutive iterations isn't worth keeping
                    #it also takes a little bit of processing but that should be okay
                    print('combining', file=sys.stderr)
                    mcDataset = mc.combineRMData([mcDataset], valueModel)[0]

                #this assumes that both player1 and player2 get requests each turn
                #which I think is accurate, but most formats will give one player a waiting request
                #this will lock up if a player causes an error, so don't do that

                async def playTurn(queue, myMcData, actionList, cmdHeader, initMoves):

                    request = await queue.get()

                    if len(initMoves) > 0:
                        action = initMoves[0]
                        del initMoves[0]
                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' pre-set action:', action, file=file)
                    else:
                        #figure out what kind of action we need
                        state = request[1]['stateHash']
                        actions = moves.getMoves(format, request[1])

                        #the mcdatasets are all combined, so we can just look at the first
                        data = myMcData[0]
                        #probs = mc.getProbsExp3(data, state, actions)
                        probs = mc.getProbsRM(data, state, actions)
                        #remove low probability moves, likely just noise
                        #this can remove every action, but if that's the case then it's doesn't really matter
                        #as all the probabilites are low
                        normProbs = np.array([p if p > probCutoff else 0 for p in probs])
                        normProbs = normProbs / np.sum(normProbs)

                        action = np.random.choice(actions, p=normProbs)

                    actionList.append(action)
                    await game.cmdQueue.put(cmdHeader + action)

                async def userTurn(queue, actionList, cmdHeader, initMoves):

                    request = await queue.get()

                    if len(initMoves) > 0:
                        action = initMoves[0]
                        del initMoves[0]
                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' pre-set action:', action, file=file)
                    else:
                        #figure out what kind of action we need
                        state = request[1]['stateHash']
                        actions = moves.getMoves(format, request[1])


                        actionTexts = []
                        for j in range(len(actions)):
                            action = actions[j].split(',')
                            actionText = []
                            for k in range(len(action)):
                                a = action[k]
                                a = a.strip()
                                if 'pass' in a:
                                    actionText.append('pass')
                                elif 'move' in a:
                                    parts = a.split(' ')
                                    moveNum = int(parts[1])
                                    if len(parts) < 3:
                                        targetNum = 0
                                    else:
                                        targetNum = int(parts[2])
                                    move = request[1]['active'][k]['moves'][moveNum-1]['move']
                                    if targetNum != 0:
                                        actionText.append(move + ' into slot ' + str(targetNum))
                                    else:
                                        actionText.append(move)
                                elif 'team' in a:
                                    actionText.append(a)
                                elif 'switch' in a:
                                    actionText.append(a)
                                else:
                                    actionText.append('unknown action: ' + a)
                            actionString = ','.join(actionText)
                            actionTexts.append(actionString)


                        #ask the user which action to take
                        print('Legal actions:')
                        for j in range(len(actions)):
                            print(j, actionTexts[j], '(' + actions[j] + ')')
                        #humans are dumb and make mistakes
                        while True:
                            try:
                                actionIndex = int(input('Your action:'))
                                if actionIndex >= 0 and actionIndex < len(actions):
                                    action = actions[actionIndex]
                                    break
                            except ValueException:
                                pass
                            print('try again')

                        actionList.append(action)

                        await game.cmdQueue.put(cmdHeader + action)


                await userTurn(game.p1Queue, p1Actions, '>p1', initMoves[0])
                await playTurn(game.p2Queue, [mcDataset[1]], p2Actions, '>p2', initMoves[1])

        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

    except Exception as e:
        print(e, file=sys.stderr)

    finally:
        mainPs.terminate()
        for ps in searchPs:
            ps.terminate()


async def main():
    await humanGame(ovoTeams, format='1v1', limit=300, numProcesses=3)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())

