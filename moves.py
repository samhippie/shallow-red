#!/usr/bin/env python3

from game import Game

def getSwitchSet(format):
    switches = []
    if format == 'singles':
        for i in range(6):
            switches.append(' switch ' + str(i+1))
    return switches

def getMoveSet(format):
    moves = []
    if format == '1v1' or format == 'singles':
        for i in range(4):
            #for extra in ['', ' mega', ' zmove']:
            for extra in ['']:
                moves.append(' move ' + str(i+1) + extra)
    return moves + getSwitchSet(format)

def getTeamSet(format):
    teams = []
    if format == '1v1':
        for i in range(3):
            teams.append(' team ' + str(i+1))
    elif format == 'singles':
        for i in range(6):
            team = []
            for j in range(6):
                mon = (i + j) % 6
                team.append(str(mon + 1))
            teams.append(' team ' + ''.join(team))
    return teams

def getWaitSet(format):
    return [' noop']

#this takes the req as a dict
#all this assumes we're playing 2v2 where you only have a team of 2
#TODO fix this so it works with singles too
def getMoves(format, req):
    if 'wait' in req:
        return [' noop']
    elif 'teamPreview' in req:
        #in restricted 2v2 there's only one choice
        #this will be expanded later
        return [' team 12']
    elif 'forceSwitch' in req:
        #impossible in 2v2
        pass
    elif 'active' in req:
        actionSets = [[], []]
        for i in [0,1]:
            actions = actionSets[i]
            mon = req['side']['pokemon'][i]
            if mon['condition'] == '0 fnt':
                actions.append(' pass')
            else:
                moves = req['active'][i]['moves']
                for j in range(len(moves)):
                    move = moves[j]
                    if move['disabled'] or move['pp'] == 0:
                        continue
                    if move['target'] == 'allySide':
                        targets = ['-1']
                    elif move['target'] == 'all' or move['target'] == 'self' or move['target'] == 'allAdjacentFoes' or move['target'] == 'allAdjacent':
                        targets = ['']
                    elif move['target'] == 'normal':
                        targets = ['-1' if i == 1 else '-2', '1', '2']
                    for target in targets:
                        actions.append('move ' + str(j+1) + ' ' + target)
        actions = []
        for move1 in actionSets[0]:
            for move2 in actionSets[1]:
                actions.append(' ' + move1 + ',' + move2)
        return actions


#easiest way to get moves for a state
"""
def getMoves(format, req):
    if req == Game.REQUEST_TEAM:
        return getTeamSet(format)

    elif req == Game.REQUEST_TURN:
        return getMoveSet(format)

    elif req == Game.REQUEST_SWITCH:
        return getSwitchSet(format)

    elif req == Game.REQUEST_WAIT:
        return getWaitSet(format)
"""
