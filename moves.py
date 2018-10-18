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
            for extra in ['', ' mega', ' zmove']:
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

#easiest way to get moves for a state
def getMoves(format, req):
    if req == Game.REQUEST_TEAM:
        return getTeamSet(format)

    elif req == Game.REQUEST_TURN:
        return getMoveSet(format)

    elif req == Game.REQUEST_SWITCH:
        return getSwitchSet(format)

    elif req == Game.REQUEST_WAIT:
        return getWaitSet(format)
