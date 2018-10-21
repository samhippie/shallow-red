#!/usr/bin/env python3

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
import montecarlo as mc

#1v1 teams in packed format
ovoTeams = [
    '|mimikyu|mimikiumz||willowisp,playrough,swordsdance,shadowsneak|Jolly|240,128,96,,,44|||||]|zygarde|groundiumz|H|thousandarrows,coil,substitute,rockslide|Impish|248,12,248,,,|||||]|volcarona|buginiumz|H|bugbuzz,quiverdance,substitute,overheat|Timid|,,52,224,,232||,0,,,,|||',

    '|naganadel|choicespecs||sludgewave,dracometeor,hiddenpowergrass,fireblast|Timid|56,,,188,64,200||,0,,,,|||]|zygarde|groundiumz|H|coil,substitute,bulldoze,thousandarrows|Impish|252,,220,,,36|||||]|magearna|fairiumz||calmmind,painsplit,irondefense,fleurcannon|Modest|224,,160,,,124||,0,,,,|||',

    '|pyukumuku|psychiumz|H|lightscreen,recover,soak,toxic|Sassy|252,,4,,252,||,0,,,,0|||]|charizard|charizarditex||flamecharge,outrage,flareblitz,swordsdance|Jolly|64,152,40,,,252|||||]|mew|keeberry||taunt,willowisp,roost,amnesia|Timid|252,,36,,,220||,0,,,,|||',

    '|tapulele|psychiumz||psychic,calmmind,reflect,moonblast|Calm|252,,60,,196,||,0,,,,|||]|charizard|charizarditex||willowisp,flamecharge,flareblitz,outrage|Jolly|252,,,,160,96|||||]|pheromosa|fightiniumz||bugbuzz,icebeam,focusblast,lunge|Modest|,,160,188,,160|||||',

    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,||,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,aurasphere|Quiet|252,4,,252,,|||||',
]

#please don't use a stall team
singlesTeams = [
    '|azelf|focussash||stealthrock,taunt,explosion,flamethrower|Jolly|,252,,4,,252|||||]|crawdaunt|focussash|H|swordsdance,knockoff,crabhammer,aquajet|Adamant|,252,,,4,252|||||]|mamoswine|focussash|H|endeavor,earthquake,iceshard,iciclecrash|Jolly|,252,,,4,252|||||]|starmie|electriumz|H|hydropump,thunder,rapidspin,icebeam|Timid|,,,252,4,252||,0,,,,|||]|scizor|ironplate|1|swordsdance,bulletpunch,knockoff,superpower|Adamant|,252,4,,,252|||||]|manectric|manectite|1|voltswitch,flamethrower,signalbeam,hiddenpowergrass|Timid|,,,252,4,252||,0,,,,|||',

    'crossy u lossy|heracross|flameorb|1|closecombat,knockoff,facade,swordsdance|Jolly|,252,,,4,252|||||]chuggy buggy|scizor|choiceband|1|bulletpunch,uturn,superpower,pursuit|Adamant|112,252,,,,144|||||]woofy woof|manectric|manectite|1|thunderbolt,voltswitch,flamethrower,hiddenpowerice|Timid|,,,252,4,252||,0,,,,|||]batzywatzy|crobat|flyiniumz|H|bravebird,defog,roost,uturn|Jolly|,252,,,4,252|||||]soggy froggy|seismitoad|leftovers|H|stealthrock,scald,earthpower,toxic|Bold|244,,252,,,12||,0,,,,|||]bitchy sissy|latias|choicescarf||dracometeor,psychic,trick,healingwish|Timid|,,,252,4,252||,0,,,,|||',
]

#restricted 2v2 doubles, with teams of 2 mons
tvtTeams = [
    '|kyogre|choicescarf||waterspout,scald,thunder,icebeam|Modest|4,,,252,,252||,0,,,,|||]|tapulele|lifeorb||protect,psychic,moonblast,allyswitch|Timid|36,,,252,,220||,0,,,,|||',
]


#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

async def playRandomGame(teams, format, ps=None):
    if not ps:
        ps = await getPSProcess()
    seed = [
        random.random() * 0x10000,
        random.random() * 0x10000,
        random.random() * 0x10000,
        random.random() * 0x10000,
    ]

    game = Game(ps, format=format, teams=teams, seed=[0,0,0,0], verbose=True)

    async def randomAgent(queue, cmdHeader):
        while True:
            req = await queue.get()
            if req[0] == Game.END:
                break
            #elif req[0] == Game.ERROR:
                #req = lastRequest
            #lastRequest = req

            actions = moves.getMoves(format, req[1])
            print(cmdHeader, 'actions', actions)

            action = random.choice(actions)
            print(cmdHeader, 'picked', action)
            await game.cmdQueue.put(cmdHeader + action)

    await game.startGame()
    gameTask = asyncio.gather(randomAgent(game.p1Queue, '>p1'),
            randomAgent(game.p2Queue, '>p2'))
    asyncio.ensure_future(gameTask)
    winner = await game.winner
    gameTask.cancel()
    print('winner:', winner)
    return winner


#plays two separately trained agents
#I just copied and pasted playTestGame and duplicated all the search functions
#so expect this to take twice as long as playTestGame with the same parameters
async def playCompGame(teams, limit1=100, limit2=100, format='1v1', numProcesses1=1, numProcesses2=1, file=sys.stdout, initMoves=([],[])):
    try:

        mainPs = await getPSProcess()

        searchPs1 = [await getPSProcess() for i in range(numProcesses1)]
        searchPs2 = [await getPSProcess() for i in range(numProcesses2)]

        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        await game.startGame()


        #holds the montecarlo data
        #each entry goes to one process
        #this will get really big with n=3 after 20ish turns if you
        #don't purge old data
        mcDatasets1 = []
        mcDatasets2 = []

        #likelihood of the search agent picking random moves
        #decreases exponentially as the search goes on
        def gamma1(iter):
            #return 1 / 2 ** (1 + 10 * iter / limit1)
            return 0.3
        def gamma2(iter):
            #return 1 / 2 ** (1 + 10 * iter / limit2)
            return 0.3
        avgGamma1 = 0.3
        avgGamma2 = 0.3
        for i in range(numProcesses1):
            mcDatasets1.append([{
                'gamma': gamma1,
                'avgGamma': avgGamma1,
            }, {
                'gamma': gamma1,
                'avgGamma': avgGamma1,
            }])
        for i in range(numProcesses2):
            mcDatasets2.append([{
                'gamma': gamma2,
                'avgGamma': avgGamma2,
            }, {
                'gamma': gamma2,
                'avgGamma': avgGamma2,
            }])

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        print('|c|server|bot1 has game limit', limit1, 'x', numProcesses1, ', bot 2 has game limit', limit2, 'x', numProcesses2, file=file)
        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            #we reassign this later, so we have to declare it nonlocal
            nonlocal mcDatasets1
            nonlocal mcDatasets2
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)

                #don't search if we aren't going to use the results
                if len(initMoves[0]) == 0 or len(initMoves[1]) == 0:
                    #I suspect that averaging two runs together will
                    #give us more improvement than running for twice as long
                    #and it should run faster than a single long search due to parallelism

                    searches1 = []
                    searches2 = []
                    for j in range(numProcesses1):
                        search1 = mc.mcSearchExp3(
                                searchPs1[j],
                                format,
                                teams,
                                limit=limit1,
                                seed=seed,
                                p1InitActions=p1Actions,
                                p2InitActions=p2Actions,
                                mcData=mcDatasets1[j])
                        searches1.append(search1)

                    for j in range(numProcesses2):
                        search2 = mc.mcSearchExp3(
                                searchPs2[j],
                                format,
                                teams,
                                limit=limit2,
                                seed=seed,
                                p1InitActions=p1Actions,
                                p2InitActions=p2Actions,
                                mcData=mcDatasets2[j])
                        searches2.append(search2)



                    await asyncio.gather(*searches1)
                    await asyncio.gather(*searches2)

                    #combine the processes results together, purge unused information
                    #this assumes that any state that isn't seen in two consecutive iterations isn't worth keeping
                    #it also takes a little bit of processing but that should be okay
                    print('combining', file=sys.stderr)

                    #record which states were seen in the last iteration
                    seenStates1 = {}
                    seenStates2 = {}
                    for data in mcDatasets1:
                        for j in range(2):
                            seen = data[j]['seenStates']
                            for state in seen:
                                seenStates1[state] = True
                    for data in mcDatasets2:
                        for j in range(2):
                            seen = data[j]['seenStates']
                            for state in seen:
                                seenStates2[state] = True


                    #combine data on states that were seen in any search
                    #in the last iteration
                    combMcData1 = [{
                        'countTable': collections.defaultdict(int),
                        'expValueTable': collections.defaultdict(int),
                        'seenStates': {},
                        'gamma': gamma1,
                        'avgGamma': avgGamma1} for j in range(2)]
                    combMcData2 = [{
                        'countTable': collections.defaultdict(int),
                        'expValueTable': collections.defaultdict(int),
                        'seenStates': {},
                        'gamma': gamma2,
                        'avgGamma': avgGamma1} for j in range(2)]
                    for data in mcDatasets1:
                        for j in range(2):
                            countTable = data[j]['countTable']
                            expValueTable = data[j]['expValueTable']
                            for state, action in countTable:
                                if state in seenStates1:
                                    combMcData1[j]['countTable'][(state, action)] += countTable[(state, action)]
                                    combMcData1[j]['expValueTable'][(state, action)] += expValueTable[(state, action)]
                    for data in mcDatasets2:
                        for j in range(2):
                            countTable = data[j]['countTable']
                            expValueTable = data[j]['expValueTable']
                            for state, action in countTable:
                                if state in seenStates2:
                                    combMcData2[j]['countTable'][(state, action)] += countTable[(state, action)]
                                    combMcData2[j]['expValueTable'][(state, action)] += expValueTable[(state, action)]

                    #copy the combined data back into the datasets
                    mcDatasets1 = [copy.deepcopy(combMcData1) for j in range(numProcesses1)]
                    mcDatasets2 = [copy.deepcopy(combMcData2) for j in range(numProcesses2)]

                #this assumes that both player1 and player2 get requests each turn
                #which I think is accurate, but most formats will give one player a waiting request
                #this will lock up if a player causes an error, so don't do that

                #TODO this signature is ugly, reorganize it so we can just pass an index
                async def playTurn(queue, myMcData, actionList, cmdHeader, initMoves):

                    request = await queue.get()

                    if len(initMoves) > 0:
                        action = initMoves[0]
                        del initMoves[0]
                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' pre-set action:', action, file=file)
                    else:
                        #figure out what kind of action we need
                        state = request[1]
                        actions = moves.getMoves(format, state[1])

                        #the mcdatasets are all combined, so we can just look at the first
                        data = myMcData[0]
                        probs = mc.getProbsExp3(data, state, actions)
                        expValue = mc.getExpValueExp3(data, state, actions, probs)
                        normProbs = np.array([p if p > probCutoff else 0 for p in probs])
                        normProbs = normProbs / np.sum(normProbs)

                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' expected value:', '%.1f%%' % (expValue * 100), file=file)
                        for j in range(len(actions)):
                            if normProbs[j] > 0:
                                print('|c|' + cmdHeader + '|Turn ' + str(i) + ' action:', actions[j],
                                        'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                        action = np.random.choice(actions, p=normProbs)

                    actionList.append(action)
                    await game.cmdQueue.put(cmdHeader + action)

                await playTurn(game.p1Queue, [data[0] for data in mcDatasets1], p1Actions, '>p1', initMoves[0])
                await playTurn(game.p2Queue, [data[1] for data in mcDatasets2], p2Actions, '>p2', initMoves[1])

        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)
        return winner

    except Exception as e:
        print(e, file=sys.stderr)

    finally:
        mainPs.terminate()
        for ps in searchPs1:
            ps.terminate()
        for ps in searchPs2:
            ps.terminate()


async def playTestGame(teams, limit=100, format='1v1', numProcesses=1, file=sys.stdout, initMoves=([],[])):
    try:

        mainPs = await getPSProcess()

        searchPs = [await getPSProcess() for i in range(numProcesses)]

        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        await game.startGame()


        #holds the montecarlo data
        #each entry goes to one process
        #this will get really big with n=3 after 20ish turns if you
        #don't purge old data
        mcDatasets = []

        def gamma(iter):
            return 0.3
        avgGamma = 0.3
        for i in range(numProcesses):
            mcDatasets.append([{
                'gamma': gamma,
                'avgGamma': avgGamma,
            }, {
                'gamma': gamma,
                'avgGamma': avgGamma,
            }])

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            #we reassign this later, so we have to declare it nonlocal
            nonlocal mcDatasets
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
                        search = mc.mcSearchExp3(
                                searchPs[j],
                                format,
                                teams,
                                limit=limit,
                                seed=seed,
                                p1InitActions=p1Actions,
                                p2InitActions=p2Actions,
                                mcData=mcDatasets[j])
                        searches.append(search)


                    await asyncio.gather(*searches)

                    #combine the processes results together, purge unused information
                    #this assumes that any state that isn't seen in two consecutive iterations isn't worth keeping
                    #it also takes a little bit of processing but that should be okay
                    print('combining', file=sys.stderr)

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
                        'gamma': gamma} for j in range(2)]
                    for data in mcDatasets:
                        for j in range(2):
                            countTable = data[j]['countTable']
                            expValueTable = data[j]['expValueTable']
                            for state, action in countTable:
                                if state in seenStates:
                                    combMcData[j]['countTable'][(state, action)] += countTable[(state, action)]
                                    combMcData[j]['expValueTable'][(state, action)] += expValueTable[(state, action)]

                    #copy the combined data back into the datasets
                    mcDatasets = [copy.deepcopy(combMcData) for j in range(numProcesses)]

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
                        probs = mc.getProbsExp3(data, state, actions)
                        expValue = mc.getExpValueExp3(data, state, actions, probs)
                        #remove low probability moves, likely just noise
                        #I'm also trying out using a temperature parameter to make it a little greedier
                        temp = 1
                        normProbs = np.array([p**(1/temp) if p > probCutoff else 0 for p in probs])
                        normProbs = normProbs / np.sum(normProbs)

                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' expected value:', '%.1f%%' % (expValue * 100), file=file)
                        for j in range(len(actions)):
                            if normProbs[j] > 0:
                                print('|c|' + cmdHeader + '|Turn ' + str(i) + ' action:', actions[j],
                                        'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                        action = np.random.choice(actions, p=normProbs)

                    actionList.append(action)
                    await game.cmdQueue.put(cmdHeader + action)

                await playTurn(game.p1Queue, [data[0] for data in mcDatasets], p1Actions, '>p1', initMoves[0])
                await playTurn(game.p2Queue, [data[1] for data in mcDatasets], p2Actions, '>p2', initMoves[1])

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

async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

async def main():
    #teams = (singlesTeams[0], singlesTeams[1])
    #teams = (ovoTeams[4], ovoTeams[4])
    teams = (tvtTeams[0], tvtTeams[0])
    initMoves = ([], [])
    #await playRandomGame(teams, format='2v2doubles')
    await playTestGame(teams, format='2v2doubles', limit=1000, numProcesses=1, initMoves = initMoves)
    """
    limit1 = 1000
    numProcesses1 = 1
    limit2 = 100
    numProcesses2 = 1
    #for i in range(100):
    #1 is temp = 1, 2 is temp = 4
    with open(os.devnull, 'w') as devnull:
        result = await playCompGame(teams, format='1v1', limit1=limit1, limit2=limit2, numProcesses1=numProcesses1, numProcesses2=numProcesses2, initMoves=initMoves, file=devnull)
        print(result)
    """


    """
    i = 0
    while True:
        limit = 100 * 2 ** i
        print('starting game with limit', limit, file=sys.stderr)
        with open('iterout' + str(limit) + '.txt', 'w') as file:
            await playTestGame(teams, format='1v1', limit=limit, numProcesses=3, file=file)
        i += 1
    """

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
