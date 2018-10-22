#!/usr/bin/env python3

import numpy as np

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

#checks if the group of mons is valid and canonical
#which means no duplicates, must be in ascending order
def checkTeamSet(set):
    counts = np.unique(set, return_counts=True)[1]
    #no dupes, sorted
    return np.all(np.unique(set, return_counts=True)[1] == 1) and np.all(np.diff(set) >= 0)

#combine two sets of pokemon into a teams
#assumes both sets are valid by checkTeamSet
def combineTeamSets(a,b):
    sets = []
    for setA in a:
        for setB in b:
            candidate = list(setA) + list(setB)
            if np.all(np.unique(candidate, return_counts=True)[1] == 1):
                sets.append(candidate)
    return sets

#makes the list of possible teams given the parameters
#up to symmetry
def makeTeams(numMons, teamSize, numInFront):
    team = list(range(1, numMons+1))
    #find leads by taking the cartesian product
    leads = np.array(np.meshgrid(*([team]*numInFront))).T.reshape(-1, numInFront)
    #and filtering
    leads = [l for l in leads if checkTeamSet(l)]
    numInBack = teamSize - numInFront
    if numInBack > 0:
        #back is found just like the leads
        back = np.array(np.meshgrid(*([team]*numInBack))).T.reshape(-1, numInBack)
        back = [b for b in back if checkTeamSet(b)]
    else:
        #there is one choice, the empty team
        back = [[]]
    teams = combineTeamSets(leads, back)
    return [' team ' + ''.join([str(t) for t in team]) for team in teams]

#maps format => teams
#teamCache = {}

#maps (format, str(req)) => actions
actionCache = {}

#using the cache seems to be a little bit faster in tests
#1000 games 1v1 went from 43s to 40s
def getMoves(format, req):
    key = (format, str(req))
    if not key in actionCache:
        actionCache[key] = getMovesImpl(format, req)
    return actionCache[key]

#this takes the req as a dict
#this works for anything that doesn't require switching
#TODO fix this so it works with regularsingles too
def getMovesImpl(format, req):
    if 'wait' in req:
        return [' noop']
    elif 'teamPreview' in req:
        #if format in teamCache:
            #return teamCache[format]
        numMons = len(req['side']['pokemon'])
        teamSize = req['maxTeamSize']
        if format == '2v2doubles':
            numInFront = 2
        else:
            numInFront = 1
        teams = makeTeams(numMons, teamSize, numInFront)
        #teamCache[format] = teams
        return teams
    elif 'forceSwitch' in req:
        #impossible in 2v2
        pass
    elif 'active' in req:
        #TODO benchmark this to see if caching helps

        #holds the possible actions for each active slot
        actionSets = []
        for i in range(len(req['active'])):
            actionSets.append([])

        for i in range(len(req['active'])):
            #go over each move, gen action for each legal target
            actions = actionSets[i]
            mon = req['side']['pokemon'][i]
            if mon['condition'] == '0 fnt':
                actions.append(' pass')
            else:
                moves = req['active'][i]['moves']
                for j in range(len(moves)):
                    move = moves[j]
                    if ('disabled' in move and move['disabled']) or ('pp' in move and move['pp'] == 0):
                        continue
                    if format != '2v2doubles' or 'target' not in move:
                        targets = ['']
                    elif move['target'] == 'allySide':
                        targets = ['-1' if i == 1 else '-2']
                    elif move['target'] in ['all', 'self', 'allAdjacentFoes', 'allAdjacent']:
                        targets = ['']
                    elif move['target'] == 'normal':
                        targets = ['-1' if i == 1 else '-2', '1', '2']
                    for target in targets:
                        actions.append('move ' + str(j+1) + ' ' + target)
        actions = []
        #cartesian product of the elements of the action sets
        actionCross = np.array(np.meshgrid(*actionSets)).T.reshape(-1, len(actionSets))
        for set in actionCross:
            actions.append(' ' + ','.join(set))
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
