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
import runner
import deepRunner





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

    #await runner.playTestGame(teams, format=format, limit=100, numProcesses=1, initMoves=initMoves, algo='cfr')#, bootstrapAlgo='rm', bootstrapPercentage=100)
    await deepRunner.playTestGame(teams, format=format, limit=100, advEpochs=1000, stratEpochs=20000, branchingLimit=3, depthLimit=5, initMoves=initMoves)

    return


    limit1 = 100
    numProcesses1 = 3
    algo1 = 'cfr'

    limit2 = 100
    numProcesses2 = 3
    algo2 = 'rm'

    with open(os.devnull, 'w') as devnull:
        result = await runner.playCompGame(teams, format=format, limit1=limit1, limit2=limit2, numProcesses1=numProcesses1, numProcesses2=numProcesses2, algo1=algo1, algo2=algo2, initMoves=initMoves, concurrent=False, file=devnull)
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
