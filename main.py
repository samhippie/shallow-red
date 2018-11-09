#!/usr/bin/env python3

import asyncio
import collections
from contextlib import suppress
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
import montecarlo.exp3 as exp3
import montecarlo.rm as rm
import montecarlo.oos as oos
import montecarlo.cfr as cfr


#putting this up top for convenience
def getAgent(algo, teams, format, valueModel=None):
    if algo == 'rm':
        agent = rm.RegretMatchAgent(
                teams=teams,
                format=format,
                posReg=True,
                probScaling=2,
                regScaling=1.5,
                valueModel=valueModel,
                verbose=False)
    elif algo == 'oos':
        agent = oos.OnlineOutcomeSamplingAgent(
                teams=teams,
                format=format,
                #posReg=True,
                #probScaling=2,
                #regScaling=1.5,
                verbose=False)
    elif algo == 'exp3':
        agent = exp3.Exp3Agent(
                teams=teams,
                format=format,
                verbose=False)
    elif algo == 'cfr':
        agent = cfr.CfrAgent(
                teams=teams,
                format=format,

                samplingType=cfr.AVERAGE,
                exploration=0.1,
                bonus=0,
                threshold=1,
                bound=3,

                posReg=False,
                probScaling=1,
                regScaling=1,

                depthLimit=3,
                evaluation=cfr.ROLLOUT,

                verbose=False)
    return agent




humanTeams = [

    '|ditto|choicescarf|H|transform|Timid|252,,4,,,252||,0,,,,|||]|zygarde|earthplate|H|thousandarrows,coil,extremespeed,bulldoze|Impish|252,252,,,4,|||||]|dragonite|lifeorb|H|extremespeed,outrage,icepunch,dragondance|Jolly|4,252,,,,252|M||||',

    '|gliscor|toxicorb|H|earthquake,toxic,substitute,protect|Jolly|252,156,,,100,|M||||]|tapulele|focussash||moonblast,psychic,calmmind,shadowball|Timid|4,,,252,,252||,0,,,,|||]|gyarados|sitrusberry||waterfall,icefang,stoneedge,dragondance|Adamant|156,252,,,,100|M||||',

]

vgcTeams = [

    '|salazzle|focussash||protect,flamethrower,sludgebomb,fakeout|Timid|,,,252,,252||||50|]|kyogre|choicescarf||waterspout,originpulse,scald,icebeam|Modest|4,,,252,,252||,0,,,,||50|]|tapulele|lifeorb||psychic,moonblast,dazzlinggleam,protect|Modest|4,,,252,,252||,0,,,,||50|]|incineroar|figyberry|H|flareblitz,knockoff,uturn,fakeout|Adamant|236,4,4,,236,28|M|||50|]|lunala|spookyplate||moongeistbeam,psyshock,tailwind,protect|Timid|4,,,252,,252||,0,,,,||50|]|toxicroak|assaultvest|1|poisonjab,lowkick,fakeout,feint|Adamant|148,108,4,,244,4|F|||50|',

]

#1v1 teams in packed format
ovoTeams = [
    '|mimikyu|mimikiumz||willowisp,playrough,swordsdance,shadowsneak|Jolly|240,128,96,,,44|||||]|zygarde|groundiumz|H|thousandarrows,coil,substitute,rockslide|Impish|248,12,248,,,|||||]|volcarona|buginiumz|H|bugbuzz,quiverdance,substitute,overheat|Timid|,,52,224,,232||,0,,,,|||',

    '|naganadel|choicespecs||sludgewave,dracometeor,hiddenpowergrass,fireblast|Timid|56,,,188,64,200||,0,,,,|||]|zygarde|groundiumz|H|coil,substitute,bulldoze,thousandarrows|Impish|252,,220,,,36|||||]|magearna|fairiumz||calmmind,painsplit,irondefense,fleurcannon|Modest|224,,160,,,124||,0,,,,|||',

    '|pyukumuku|psychiumz|H|lightscreen,recover,soak,toxic|Sassy|252,,4,,252,||,0,,,,0|||]|charizard|charizarditex||flamecharge,outrage,flareblitz,swordsdance|Jolly|64,152,40,,,252|||||]|mew|keeberry||taunt,willowisp,roost,amnesia|Timid|252,,36,,,220||,0,,,,|||',

    '|tapulele|psychiumz||psychic,calmmind,reflect,moonblast|Calm|252,,60,,196,||,0,,,,|||]|charizard|charizarditex||willowisp,flamecharge,flareblitz,outrage|Jolly|252,,,,160,96|||||]|pheromosa|fightiniumz||bugbuzz,icebeam,focusblast,lunge|Modest|,,160,188,,160|||||',

    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|M||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,|M|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,freezedry|Quiet|252,4,,252,,|M||||',

    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|M||||]|bulbasaur|chestoberry||fissure,toxic,sludgebomb,rest|Quiet|252,4,,252,,|M|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,freezedry|Quiet|252,4,,252,,|M||||',
]

#please don't use a stall team
singlesTeams = [
    '|azelf|focussash||stealthrock,taunt,explosion,flamethrower|Jolly|,252,,4,,252|||||]|crawdaunt|focussash|H|swordsdance,knockoff,crabhammer,aquajet|Adamant|,252,,,4,252|||||]|mamoswine|focussash|H|endeavor,earthquake,iceshard,iciclecrash|Jolly|,252,,,4,252|||||]|starmie|electriumz|H|hydropump,thunder,rapidspin,icebeam|Timid|,,,252,4,252||,0,,,,|||]|scizor|ironplate|1|swordsdance,bulletpunch,knockoff,superpower|Adamant|,252,4,,,252|||||]|manectric|manectite|1|voltswitch,flamethrower,signalbeam,hiddenpowergrass|Timid|,,,252,4,252||,0,,,,|||',

    'crossy u lossy|heracross|flameorb|1|closecombat,knockoff,facade,swordsdance|Jolly|,252,,,4,252|||||]chuggy buggy|scizor|choiceband|1|bulletpunch,uturn,superpower,pursuit|Adamant|112,252,,,,144|||||]woofy woof|manectric|manectite|1|thunderbolt,voltswitch,flamethrower,hiddenpowerice|Timid|,,,252,4,252||,0,,,,|||]batzywatzy|crobat|flyiniumz|H|bravebird,defog,roost,uturn|Jolly|,252,,,4,252|||||]soggy froggy|seismitoad|leftovers|H|stealthrock,scald,earthpower,toxic|Bold|244,,252,,,12||,0,,,,|||]bitchy sissy|latias|choicescarf||dracometeor,psychic,trick,healingwish|Timid|,,,252,4,252||,0,,,,|||',
]

#restricted 2v2 doubles, with teams of 2 mons
tvtTeams = [
    '|kyogre|choicescarf||waterspout,scald,thunder,icebeam|Modest|4,,,252,,252||,0,,,,|||]|tapulele|lifeorb||protect,psychic,moonblast,allyswitch|Timid|36,,,252,,220||,0,,,,|||',

    'I hate anime|tapufini|choicescarf||muddywater,dazzlinggleam,haze,icebeam|Modest|12,,,252,,244||,0,,,,||50|]Anime is life|salazzle|focussash||fakeout,sludgebomb,flamethrower,protect|Timid|4,,,252,,252||||50|',

    'can\'t trust others|garchomp|groundiumz|H|earthquake,rockslide,substitute,protect|Adamant|12,156,4,,116,220|M|||50|]dirty dan|mukalola|aguavberry|1|knockoff,poisonjab,shadowsneak,protect|Adamant|188,244,44,,20,12|M|||50|',

    '|venusaur|focussash|H|protect,sludgebomb,grassknot,sleeppowder|Modest|4,,,252,,252|M|,0,,,,||50|]|groudon|figyberry||precipiceblades,rockslide,swordsdance,protect|Jolly|116,252,,,,140||||50|',

    '|lunala|spookyplate||moongeistbeam,psyshock,psychup,protect|Timid|4,,,252,,252||,0,,,,||50|]|incineroar|figyberry|H|flareblitz,knockoff,uturn,fakeout|Adamant|236,4,4,,236,28|M|||50|',

    '|kartana|assaultvest||leafblade,smartstrike,sacredsword,nightslash|Jolly|204,4,4,,84,212||||50|]|tapukoko|choicespecs||thunderbolt,dazzlinggleam,discharge,voltswitch|Timid|140,,36,204,28,100||,0,,,,||50|',

    '|tapulele|choicespecs||thunderbolt,psychic,moonblast,dazzlinggleam|Modest|252,,,252,,||,0,,,,||50|]|kartana|focussash||detect,leafblade,sacredsword,smartstrike|Jolly|,252,,,4,252||||50|'
]


#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'


async def playRandomGame(teams, format, ps=None, initMoves=[[],[]], seed=None):
    if not ps:
        ps = await getPSProcess()
    if not seed:
        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

    game = Game(ps, format=format, teams=teams, seed=seed, verbose=True)

    async def randomAgent(queue, cmdHeader, initMoveList):
        while True:
            req = await queue.get()
            if req[0] == Game.END:
                break

            #print('getting actions')
            actions = moves.getMoves(format, req[1])
            state = req[1]['state']
            #print(cmdHeader, 'actions', actions)

            if len(initMoveList) > 0:
                action = initMoveList[0]
                del initMoveList[0]
            else:
                action = random.choice(actions)
            print(cmdHeader, 'picked', action)
            await game.cmdQueue.put(cmdHeader + action)

    await game.startGame()
    gameTask = asyncio.gather(randomAgent(game.p1Queue, '>p1', initMoves[0]),
            randomAgent(game.p2Queue, '>p2', initMoves[1]))
    asyncio.ensure_future(gameTask)
    winner = await game.winner
    gameTask.cancel()
    print('winner:', winner)
    return winner


#plays two separately trained agents
#I just copied and pasted playTestGame and duplicated all the search functions
#so expect this to take twice as long as playTestGame with the same parameters
async def playCompGame(teams, limit1=100, limit2=100, time1=None, time2=None, format='1v1', numProcesses1=1, numProcesses2=1, algo1='rm', algo2='rm', file=sys.stdout, initMoves=([],[]), valueModel1=None, valueModel2=None, concurrent=False):
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

        if time1:
            limit1 = 100000000
        if time2:
            limit2 = 100000000

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        agent1 = getAgent(algo1, teams, format, valueModel1)
        agent2 = getAgent(algo2, teams, format, valueModel2)

        await game.startGame()

        #moves with probabilites below this are not considered
        probCutoff = 0.03

        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            nonlocal searchPs1
            nonlocal searchPs2
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)

                #don't search if we aren't going to use the results
                if len(initMoves[0]) == 0 or len(initMoves[1]) == 0:

                    searches1 = []
                    searches2 = []
                    for j in range(numProcesses1):
                        search1 = agent1.search(
                                ps=searchPs1[j],
                                pid=j,
                                limit=limit1,
                                seed=seed)
                        searches1.append(search1)

                    for j in range(numProcesses2):
                        search2 = agent2.search(
                                ps=searchPs2[j],
                                pid=j,
                                limit=limit2,
                                seed=seed)
                        searches2.append(search2)

                    if concurrent and time1 == None and time2 == None:
                        await asyncio.gather(*searches1, *searches2)
                    elif time1 and time2:
                        #there's a timeout function, but I got this working first
                        searchTask = asyncio.ensure_future(asyncio.gather(*searches1))
                        await asyncio.sleep(time1)
                        searchTask.cancel()
                        with suppress(asyncio.CancelledError):
                            await searchTask

                        searchTask = asyncio.ensure_future(asyncio.gather(*searches2))
                        await asyncio.sleep(time2)
                        searchTask.cancel()
                        with suppress(asyncio.CancelledError):
                            await searchTask

                        #restart the search processes just to clean things up
                        #for ps in searchPs1 + searchPs2:
                            #ps.terminate
                        #searchPs1 = [await getPSProcess() for i in range(numProcesses1)]
                        #searchPs2 = [await getPSProcess() for i in range(numProcesses2)]

                    else:
                        await asyncio.gather(*searches1)
                        await asyncio.gather(*searches2)

                    #restart the search processes just to clean things up
                    for ps in searchPs1 + searchPs2:
                        ps.terminate()
                    searchPs1 = [await getPSProcess() for i in range(numProcesses1)]
                    searchPs2 = [await getPSProcess() for i in range(numProcesses2)]

                    #let the agents combine and purge data
                    print('combining', file=sys.stderr)
                    agent1.combine()
                    agent2.combine()


                #player-specific
                queues = [game.p1Queue, game.p2Queue]
                actionLists = [p1Actions, p2Actions]
                cmdHeaders = ['>p1', '>p2']
                agents = [agent1, agent2]

                async def playTurn(num):

                    request = await queues[num].get()

                    if len(initMoves[num]) > 0:
                        #do the given action
                        action = initMoves[num][0]
                        del initMoves[num][0]
                        print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' pre-set action:', action, file=file)
                    else:
                        #let the agent pick the action
                        #figure out what kind of action we need
                        state = request[1]['stateHash']
                        actions = moves.getMoves(format, request[1])

                        probs = agents[num].getProbs(num, state, actions)
                        #remove low probability moves, likely just noise
                        normProbs = np.array([p if p > probCutoff else 0 for p in probs])
                        normSum = np.sum(normProbs)
                        if normSum > 0:
                            normProbs = normProbs / np.sum(normProbs)
                        else:
                            normProbs = [1 / len(actions) for a in actions]

                        for j in range(len(actions)):
                            actionString = moves.prettyPrintMove(actions[j], request[1])
                            if normProbs[j] > 0:
                                print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' action:', actionString,
                                        'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                        action = np.random.choice(actions, p=normProbs)

                    actionLists[num].append(action)
                    await game.cmdQueue.put(cmdHeaders[num] + action)

                await playTurn(0)
                await playTurn(1)

        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)
        return winner

    except:
        raise

    finally:
        mainPs.terminate()
        for ps in searchPs1:
            ps.terminate()
        for ps in searchPs2:
            ps.terminate()


#trains a model for the given format with the given teams
#returns the trained model
async def trainModel(teams, format, games=100, epochs=100, numProcesses=1, valueModel=None, saveDir=None, saveName=None):
    try:
        searchPs = [await getPSProcess() for p in range(numProcesses)]

        if not valueModel:
            valueModel = model.TrainedModel(alpha=0.0001)

        agent = getAgent('rm', teams, format, valueModel=valueModel)

        print('starting network training', file=sys.stderr)
        for i in range(epochs):
            print('epoch', i, 'running', file=sys.stderr)
            valueModel.t = i / epochs

            searches = []
            for j in range(numProcesses):
                search = agent.search(
                    ps=searchPs[j],
                    pid=j,
                    limit=games)
                searches.append(search)
            await asyncio.gather(*searches)

            print('epoch', i, 'training', file=sys.stderr)
            valueModel.train(epochs=10)
            if saveDir and saveName:
                valueModel.saveModel(saveDir, saveName)

    except:
        raise

    finally:
        for ps in searchPs:
            ps.terminate()
    return valueModel


async def playTestGame(teams, limit=100, time=None, format='1v1', seed=None, numProcesses=1, valueModel=None, algo='rm', file=sys.stdout, initMoves=([],[])):
    try:

        mainPs = await getPSProcess()

        searchPs = [await getPSProcess() for i in range(numProcesses)]

        if not seed:
            seed = [
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
                random.random() * 0x10000,
            ]

        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        agent = getAgent(algo, teams, format, valueModel)

        if time:
            limit = 100000

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
            #nonlocal mcDataset
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)

                #don't search if we aren't going to use the results
                if len(initMoves[0]) == 0 or len(initMoves[1]) == 0:

                    searches = []
                    for j in range(numProcesses):
                        search = agent.search(
                                ps=searchPs[j],
                                pid=j,
                                limit=limit,
                                seed=seed,
                                initActions=[p1Actions, p2Actions])
                        searches.append(search)


                    #there's a timeout function, but I got this working first
                    if time:
                        searchTask = asyncio.ensure_future(asyncio.gather(*searches))
                        await asyncio.sleep(time)
                        searchTask.cancel()
                        with suppress(asyncio.CancelledError):
                            await searchTask

                    else:
                        await asyncio.gather(*searches)


                    #let the agents combine and purge data
                    print('combining', file=sys.stderr)
                    agent.combine()

                #player-specific
                queues = [game.p1Queue, game.p2Queue]
                actionLists = [p1Actions, p2Actions]
                cmdHeaders = ['>p1', '>p2']

                async def playTurn(num):

                    request = await queues[num].get()

                    if len(initMoves[num]) > 0:
                        #do the given action
                        action = initMoves[num][0]
                        del initMoves[num][0]
                        print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' pre-set action:', action, file=file)
                    else:
                        #let the agent pick the action
                        #figure out what kind of action we need
                        state = request[1]['stateHash']
                        actions = moves.getMoves(format, request[1])

                        probs = agent.getProbs(num, state, actions)
                        #remove low probability moves, likely just noise
                        normProbs = np.array([p if p > probCutoff else 0 for p in probs])
                        normSum = np.sum(normProbs)
                        if normSum > 0:
                            normProbs = normProbs / np.sum(normProbs)
                        else:
                            normProbs = [1 / len(actions) for a in actions]

                        for j in range(len(actions)):
                            actionString = moves.prettyPrintMove(actions[j], request[1])
                            if normProbs[j] > 0:
                                print('|c|' + cmdHeaders[num] + '|Turn ' + str(i) + ' action:', actionString,
                                        'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                        action = np.random.choice(actions, p=normProbs)

                    actionLists[num].append(action)
                    await game.cmdQueue.put(cmdHeaders[num] + action)

                await playTurn(0)
                await playTurn(1)


        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

    except:
        raise

    finally:
        mainPs.terminate()
        for ps in searchPs:
            ps.terminate()


async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

async def main():
    format = '1v1'
    #format = '2v2doubles'
    #format='singles'
    #format='vgc'

    #teams = (singlesTeams[0], singlesTeams[1])
    #gen 1 starters mirror
    teams = (ovoTeams[4], ovoTeams[4])

    #groudon vs lunala vgv19
    #teams = (tvtTeams[3], tvtTeams[4])
    #fini vs koko vgc17
    #teams = (tvtTeams[1], tvtTeams[5])

    #vgc19 scarf kyogre mirror
    #teams = (vgcTeams[0], vgcTeams[0])

    #lunala mirror
    #teams = (tvtTeams[4], tvtTeams[4])

    #salazzle/fini vs lele/kartana singles
    #teams = (tvtTeams[1], tvtTeams[6])
    #initMoves = ([' team 21'], [' team 12'])

    #initMoves = ([' team 12'], [' team 12'])
    #initMoves = ([' team 1'], [' team 1'])
    initMoves = ([], [])

    #teams = (ovoTeams[5], ovoTeams[5])
    #initMoves = ([' team 2'], [' team 2'])

    """
    ps = await getPSProcess()
    try:
        await playRandomGame(teams, format=format, ps=ps)
    finally:
        ps.terminate()
    """


    #saveDir = '/home/sam/scratch/psbot/models'
    #saveName = 'fini'
    #valueModel = model.TrainedModel(alpha=0.001)
    #await trainModel(teams=teams, format=format, games=100, epochs=10000, numProcesses=3, valueModel=valueModel, saveDir=saveDir, saveName=saveName)
    #valueModel.saveModel(saveDir, saveName)

    #valueModel = model.TrainedModel()
    #valueModel.loadModel(saveDir, saveName)
    #valueModel.training = False

    #await playTestGame(teams, format=format, limit=100, numProcesses=3, initMoves=initMoves, algo='cfr')

    #return


    limit1 = 100
    numProcesses1 = 3
    algo1 = 'cfr'

    limit2 = 100
    numProcesses2 = 3
    algo2 = 'rm'

    with open(os.devnull, 'w') as devnull:
        result = await playCompGame(teams, format=format, limit1=limit1, limit2=limit2, numProcesses1=numProcesses1, numProcesses2=numProcesses2, algo1=algo1, algo2=algo2, initMoves=initMoves, concurrent=False, file=devnull)
        print(result, flush=True)

    """
    i = 0
    while True:
        limit = 1000 * 2 ** i
        print('starting game with limit', limit, file=sys.stderr)
        with open('iterout' + str(i) + '.txt', 'w') as file:
            await playTestGame(teams, format='2v2doubles', limit=limit, valueModel=valueModel, numProcesses=1, initMoves=initMoves, file=file)
        i += 1
    """

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
