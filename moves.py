#!/usr/bin/env python3

import numpy as np
import sys

from game import Game

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

doublesFormats = ['doubles', '2v2doubles', '2v2', 'vgc']

#using the cache seems to be a little bit faster in tests
#1000 games 1v1 went from 43s to 40s
def getMoves(format, req):
    key = (format, str(req))
    if not key in actionCache:
        actionCache[key] = getMovesImpl(format, req)
    return actionCache[key]

#this takes the req as a dict
#this works for anything that doesn't require switching
def getMovesImpl(format, req):
    if 'wait' in req:
        return [' noop']
    elif 'teamPreview' in req:
        numMons = len(req['side']['pokemon'])
        #can only bring however many mons you have
        teamSize = min(req['maxTeamSize'], numMons)
        if format in doublesFormats:
            numInFront = 2
        else:
            numInFront = 1
        teams = makeTeams(numMons, teamSize, numInFront)
        #teamCache[format] = teams
        return teams
    elif 'forceSwitch' in req:
        #holds the possible actions for each active slot
        actionSets = []
        for i in range(len(req['forceSwitch'])):
            actions = []
            actionSets.append(actions)
            if not req['forceSwitch'][i]:
                actions.append('pass')
            else:
                #pick the possible switching targets
                for j in range(len(req['side']['pokemon'])):
                    mon = req['side']['pokemon'][j]
                    if not mon['active'] and not mon['condition'] == '0 fnt':
                        actions.append('switch ' + str(j+1))
        actions = []
        #cartesian product of the elements of the action sets
        actionCross = np.array(np.meshgrid(*actionSets)).T.reshape(-1, len(actionSets))
        for set in actionCross:
            #check if multiple actions switch to the same mon
            switchTargets = [int(a.split(' ')[1]) for a in set if 'switch' in a]
            _,  counts = np.unique(switchTargets, return_counts=True)
            if not any(counts > 1):
                actions.append(' ' + ','.join(set))
            elif len(actionCross) == 1:#only one mon left
                #need to pass some of the switches
                #the proper way is to replace all duplicates with a pass
                #I'm just going to hard code this for doubles
                actions.append(' ' + set[0] + ',pass')
            # else: it's just an illegal action
        return actions

    elif 'active' in req:
        #holds the possible actions for each active slot
        actionSets = []
        for i in range(len(req['active'])):
            actionSets.append([])

        for i in range(len(req['active'])):
            #go over each move, gen action for each legal target
            actions = actionSets[i]
            mon = req['side']['pokemon'][i]
            if mon['condition'] == '0 fnt':
                actions.append('pass')
            else:
                moves = req['active'][i]['moves']
                for j in range(len(moves)):
                    move = moves[j]
                    if ('disabled' in move and move['disabled']) or ('pp' in move and move['pp'] == 0):
                        continue
                    if format not in doublesFormats or 'target' not in move:
                        targets = ['']
                    #elif move['target'] == 'allySide':
                        #targets = ['-1' if i == 1 else '-2']
                    elif move['target'] in ['all', 'self', 'allAdjacentFoes', 'allAdjacent', 'randomNormal', 'foeSide', 'allySide']:
                        targets = ['']
                    elif move['target'] in ['normal', 'any']:
                        targets = ['-1' if i == 1 else '-2', '1', '2']
                    for target in targets:
                        actions.append('move ' + str(j+1) + ' ' + target)

            #pick the possible switching targets
            for j in range(len(req['side']['pokemon'])):
                mon = req['side']['pokemon'][j]
                if not mon['active'] and not mon['condition'] == '0 fnt':
                    actions.append('switch ' + str(j+1))


        actions = []
        #cartesian product of the elements of the action sets
        actionCross = np.array(np.meshgrid(*actionSets)).T.reshape(-1, len(actionSets))
        for set in actionCross:
            #check if multiple actions switch to the same mon
            switchTargets = [int(a.split(' ')[1]) for a in set if 'switch' in a]
            _,  counts = np.unique(switchTargets, return_counts=True)
            if not any(counts > 1):
                actions.append(' ' + ','.join(set))
        return actions


