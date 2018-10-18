#!/usr/bin/env python3

import sys
import subprocess
import asyncio
import random
import collections
import math
import numpy as np

import moves
from game import Game
import montecarlo as mc

#1v1 teams in packed format
testTeams = [
    '|mimikyu|mimikiumz||willowisp,playrough,swordsdance,shadowsneak|Jolly|240,128,96,,,44|||||]|zygarde|groundiumz|H|thousandarrows,coil,substitute,rockslide|Impish|248,12,248,,,|||||]|volcarona|buginiumz|H|bugbuzz,quiverdance,substitute,overheat|Timid|,,52,224,,232||,0,,,,|||',

    '|naganadel|choicespecs||sludgewave,dracometeor,hiddenpowergrass,fireblast|Timid|56,,,188,64,200||,0,,,,|||]|zygarde|groundiumz|H|coil,substitute,bulldoze,thousandarrows|Impish|252,,220,,,36|||||]|magearna|fairiumz||calmmind,painsplit,irondefense,fleurcannon|Modest|224,,160,,,124||,0,,,,|||',

    '|pyukumuku|psychiumz|H|lightscreen,recover,soak,toxic|Sassy|252,,4,,252,||,0,,,,0|||]|charizard|charizarditex||flamecharge,outrage,flareblitz,swordsdance|Jolly|64,152,40,,,252|||||]|mew|keeberry||taunt,willowisp,roost,amnesia|Timid|252,,36,,,220||,0,,,,|||',

    '|tapulele|psychiumz||psychic,calmmind,reflect,moonblast|Calm|252,,60,,196,||,0,,,,|||]|charizard|charizarditex||willowisp,flamecharge,flareblitz,outrage|Jolly|252,,,,160,96|||||]|pheromosa|fightiniumz||bugbuzz,icebeam,focusblast,lunge|Modest|,,160,188,,160|||||',

    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,||,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,aurasphere|Quiet|252,4,,252,,|||||',
]

#please don't use a stall team
testSinglesTeams = [
    '|azelf|focussash||stealthrock,taunt,explosion,flamethrower|Jolly|,252,,4,,252|||||]|crawdaunt|focussash|H|swordsdance,knockoff,crabhammer,aquajet|Adamant|,252,,,4,252|||||]|mamoswine|focussash|H|endeavor,earthquake,iceshard,iciclecrash|Jolly|,252,,,4,252|||||]|starmie|electriumz|H|hydropump,thunder,rapidspin,icebeam|Timid|,,,252,4,252||,0,,,,|||]|scizor|ironplate|1|swordsdance,bulletpunch,knockoff,superpower|Adamant|,252,4,,,252|||||]|manectric|manectite|1|voltswitch,flamethrower,signalbeam,hiddenpowergrass|Timid|,,,252,4,252||,0,,,,|||',

    'crossy u lossy|heracross|flameorb|1|closecombat,knockoff,facade,swordsdance|Jolly|,252,,,4,252|||||]chuggy buggy|scizor|choiceband|1|bulletpunch,uturn,superpower,pursuit|Adamant|112,252,,,,144|||||]woofy woof|manectric|manectite|1|thunderbolt,voltswitch,flamethrower,hiddenpowerice|Timid|,,,252,4,252||,0,,,,|||]batzywatzy|crobat|flyiniumz|H|bravebird,defog,roost,uturn|Jolly|,252,,,4,252|||||]soggy froggy|seismitoad|leftovers|H|stealthrock,scald,earthpower,toxic|Bold|244,,252,,,12||,0,,,,|||]bitchy sissy|latias|choicescarf||dracometeor,psychic,trick,healingwish|Timid|,,,252,4,252||,0,,,,|||',
]


#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

async def playRandomGame():
    ps = await getPSProcess()
    try:
        random.seed(0)
        game = Game(ps, testTeams, seed=[0,0,0,0], verbose=True)

        async def randomAgent(queue, cmdHeader):
            lastRequest = None
            while True:
                req = await queue.get()
                print(cmdHeader, queue)
                if req[0] == Game.END:
                    break
                elif req[0] == Game.ERROR:
                    req = lastRequest
                lastRequest = req

                if req[1][1] == Game.REQUEST_TEAM:
                    actions = moves.teamSet
                elif req[1][1] == Game.REQUEST_TURN:
                    actions = moves.moveSet

                action = random.choice(actions)
                await game.cmdQueue.put(cmdHeader + action)

        await game.startGame()
        await asyncio.gather(randomAgent(game.p1Queue, '>p1'),
                randomAgent(game.p2Queue, '>p2'))

        winner = await game.winner
        print('winner:', winner)
    except Exception as e:
        print(e)
    finally:
        ps.terminate()



async def playTestGame(teams, limit=100, format='1v1', file=sys.stdout):
    try:
        numProcesses = 3

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


        #this needs to be a coroutine so we can cancel it when the game ends
        #which due to concurrency issues might not be until we get into the MCTS loop
        async def play():
            i = 0
            #actions taken so far by in the actual game
            p1Actions = []
            p2Actions = []
            while True:
                i += 1
                print('starting turn', i, file=sys.stderr)



                #holds the montecarlo data
                #each entry goes to one process
                #I ran out of RAM keeping this for each turn, so I'm going to restart it each time
                mcData = [[{}, {}]] * numProcesses

                #I suspect that averaging two runs together will
                #give us more improvement than running for twice as long
                #and it should run faster than a single long search due to parallelism

                #advance both prob tables
                searches = [ mc.mcSearchExp3(
                            searchPs[i],
                            format,
                            teams,
                            limit=limit,
                            seed=seed,
                            p1InitActions=p1Actions,
                            p2InitActions=p2Actions,
                            mcData=mcData[i])
                            for i in range(numProcesses) ]

                await asyncio.gather(*searches)

                #this assumes that both player1 and player2 get requests each turn
                #which I think is accurate, but most formats will give one player a waiting request
                #except for errors

                async def playTurn(queue, myMcData, actionList, cmdHeader):
                    #figure out what kind of action we need
                    request = await queue.get()
                    state = request[1]
                    actions = moves.getMoves(format, state[1])

                    normProbList = []
                    expValueList = []
                    for data in myMcData:
                        normProbs = mc.getProbsExp3(data, state, actions)
                        expValue = mc.getExpValueExp3(data, state, actions, normProbs)
                        normProbList.append(normProbs)
                        expValueList.append(expValue)

                    expValue = np.mean(expValueList)
                    normProbs = np.mean(normProbList, axis=0)
                    probSum = np.sum(normProbs)
                    print('|c|' + cmdHeader + '|Turn ' + str(i) + ' expected value:', '%.1f%%' % (expValue * 100), file=file)
                    for j in range(len(actions)):
                        if normProbs[j] > 0:
                            print('|c|' + cmdHeader + '|Turn ' + str(i) + ' action:', actions[j],
                                    'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                    action = np.random.choice(actions, p=normProbs)
                    actionList.append(action)
                    await game.cmdQueue.put(cmdHeader + action)

                await playTurn(game.p1Queue, [data[0] for data in mcData], p1Actions, '>p1')
                await playTurn(game.p2Queue, [data[1] for data in mcData], p2Actions, '>p2')

        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

    except Exception as e:
        print(e)

    finally:
        mainPs.terminate()
        searchPs1.terminate()
        searchPs2.terminate()

async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

async def main():
    teams = (testSinglesTeams[0], testSinglesTeams[1])
    #teams = (testTeams[0], testTeams[3])
    await playTestGame(teams, format='singles', limit=1000)
    """
    i = 0
    while True:
        limit = 100 * 2 ** i
        print('starting game with limit', limit, file=sys.stderr)
        with open('iterout' + str(limit) + '.txt', 'w') as file:
            await playTestGame(limit=limit, file=file)
        i += 1
    """

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
