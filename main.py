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
    '|azelf|focussash||stealthrock,taunt,explosion,flamethrower|Jolly|,252,,4,,252|||||]|crawdaunt|focussash|H|swordsdance,knockoff,crabhammer,aquajet|Adamant|,252,,,4,252|||||]|mamoswine|focussash|H|endeavor,earthquake,iceshard,iciclecrash|Jolly|,252,,,4,252|||||]|starmie|electriumz|H|hydropump,thunder,rapidspin,icebeam|Timid|,,,252,4,252||,0,,,,|||]|scizor|ironplate|1|swordsdance,bulletpunch,knockoff,superpower|Adamant|,252,4,,,252|||||]|manectricmega|manectite|lightningrod|voltswitch,flamethrower,signalbeam,hiddenpowergrass|Timid|,,,252,4,252||,0,,,,|||',

    'crossy u lossy|heracross|flameorb|1|closecombat,knockoff,facade,swordsdance|Jolly|,252,,,4,252|||||]chuggy buggy|scizor|choiceband|1|bulletpunch,uturn,superpower,pursuit|Adamant|112,252,,,,144|||||]woofy woof|manectricmega|manectite||thunderbolt,voltswitch,flamethrower,hiddenpowerice|Timid|,,,252,4,252||,0,,,,|||]batzywatzy|crobat|flyiniumz|H|bravebird,defog,roost,uturn|Jolly|,252,,,4,252|||||]soggy froggy|seismitoad|leftovers|H|stealthrock,scald,earthpower,toxic|Bold|244,,252,,,12||,0,,,,|||]bitchy sissy|latias|choicescarf||dracometeor,psychic,trick,healingwish|Timid|,,,252,4,252||,0,,,,|||',
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



async def playTestGame(limit=100, file=sys.stdout):
    try:
        mainPs = await getPSProcess()
        searchPs = await getPSProcess()

        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

        #teams = (testTeams[4], testTeams[4])
        teams = (testSinglesTeams[0], testSinglesTeams[1])
        format = 'singles'
        game = Game(mainPs, format=format, teams=teams, seed=seed, verbose=True, file=file)

        await game.startGame()

        #holds the montecarlo data
        mcData = [{}, {}]

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
                #advance both prob tables
                await mc.mcSearchExp3(searchPs,
                        format,
                        teams,
                        limit=limit,
                        seed=seed,
                        p1InitActions=p1Actions,
                        p2InitActions=p2Actions,
                        mcData=mcData)

                #this assumes that both player1 and player2 get requests each turn
                #which I think is accurate, but most formats will give one player a waiting request
                #except for errors

                #temps above 1 are exploitative
                #temps below 1 are explorative (which isn't what we're looking for)
                temp=0.1

                async def playTurn(queue, myMcData, actionList, cmdHeader):
                    #figure out what kind of action we need
                    request = await queue.get()
                    state = request[1]
                    actions = moves.getMoves(format, state[1])
                    #if state[1] == Game.REQUEST_TEAM:
                        #actions = moves.getTeamSet(format)
                    #elif state[1] == Game.REQUEST_TURN:
                        #actions = moves.getMoveSet(format)
                    #elif state[1] == GAME.REQUEST_SWITCH:
                        #actions = moves.getSwitchSet(format)
                    #elif state[1] == GAME.REQUEST_WAIT:
                        #actions = moves.getWaitSet(format)
                    """
                    #get the probability of each action winning
                    probs = [win / (max(count, 1)) for win,count in [probTable[(request[1], action)] for action in actions]]
                    #apply temperature parameter to make the agent greedier
                    #also zero out probs < 0, which are illegal moves
                    probsTemp = np.array([p ** (1 / temp) if p >= 0 else 0 for p in probs])
                    #make it sum to 1 so np likes it
                    probSum = np.sum(probsTemp)
                    normProbs = probsTemp / probSum
                    """


                    normProbs = mc.getProbsExp3(myMcData, state, actions)
                    probSum = np.sum(normProbs)
                    expValue = mc.getExpValueExp3(myMcData, state, actions, normProbs)
                    if probSum == 0:
                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' seems impossible to play', file=file)
                    else:
                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' expected value:', '%.1f%%' % (expValue * 100), file=file)
                        for j in range(len(actions)):
                            if normProbs[j] > 0:
                                print('|c|' + cmdHeader + '|Turn ' + str(i) + ' action:', actions[j],
                                        'prob:', '%.1f%%' % (normProbs[j] * 100), file=file)

                    #pick according to the probability (or should we be 100% greedy?)
                    action = np.random.choice(actions, p=normProbs)
                    actionList.append(action)
                    await game.cmdQueue.put(cmdHeader + action)

                await playTurn(game.p1Queue, mcData[0], p1Actions, '>p1')
                await playTurn(game.p2Queue, mcData[1], p2Actions, '>p2')

        #gameTask = play()
        #asyncio.ensure_future(gameTask)
        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

    except Exception as e:
        print(e)

    finally:
        mainPs.terminate()
        searchPs.terminate()

async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

async def main():
    await playTestGame(limit=100)
    """
    i = 0
    while True:
        limit = 100 * 2 ** i
        print('starting game with limit', limit, file=sys.stderr)
        with open('iterout' + str(limit) + '.txt', 'w') as file:
            await playTestGame(limit=limit, file=file)
        i += 1
    """
    #await playRandomGame()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
