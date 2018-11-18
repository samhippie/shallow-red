#!/usr/bin/env python3

import numpy as np

#this file is a bit messy as both value models and deep cfr models
#both use these methods

stateSize = 3883

#3883 is the state size
#186 is the action size
inputShape = (stateSize + 2 * 186,)

#number of possible actions, which is used for our enumeration
numActions = 2486

#used to store the maps from id,type to enumeration
idMap = {}

#gets a sequential index for the given id of the given type
#generating one if necessary
def enumId(type, id):
    #get mapping for type
    if not type in idMap:
        idMap[type] = {
            'nextNum': 0
        }
    typeMap = idMap[type]

    #get enumerated index within the type
    if not id in typeMap:
        num = typeMap['nextNum']
        typeMap['nextNum'] += 1
        typeMap[id] = num
    return typeMap[id]

#turns the state into a tensor (really just an array)
#structure should be constant across runs
#as long as idMap stays the same
def stateToTensor(state):
    stateTensor = np.zeros(0)
    #weather section
    num = 5
    size = 11 # duration 0-10
    weatherArray = np.zeros(num * size)
    weather = state['weather']
    for id in weather:
        n = enumId('weather', id)
        duration = weather[id]
        oneHot = numToOneHot(duration, size)
        insertSublist(weatherArray, n, oneHot)

    stateTensor = np.concatenate([stateTensor, weatherArray])

    #player section
    for i in range(2):
        player = state['players'][i]
        playerList = np.zeros(0)
        #zmove
        zMove = numToOneHot(player['zMoveUsed'], 2)
        playerList = np.concatenate([playerList, zMove])

        #mons
        num = 6
        #active only, all 0s for non active:
        #newly switched (2), ability (24), addedType (13), move status (12), boosts (7 * 13), volatiles (10 * 10)
        activeSize = 2 + 24 + 13 + 12 + (7 * 13) + (10 * 10)
        #for all:
        #is active (2), details (24), status (10), hp (11), item (13)
        size = 2 + 24 + 10 + 11 + 13 + activeSize
        monList = np.zeros(num * size)
        for monId in player['mons']:
            mon = player['mons'][monId]
            monPlace = enumId('mon-id', monId)

            #go through each field
            sublists = []
            #non-active first
            isActive = monId in player['active']
            sublists.append(numToOneHot(isActive, 2))

            details = enumId('mon-details', mon['details'])
            sublists.append(numToOneHot(details, 24))

            status = enumId('mon-status', mon['status'])
            sublists.append(numToOneHot(status, 10))

            hp = mon['hp']
            sublists.append(numToOneHot(hp, 11))

            item = enumId('mon-item', mon['item'])
            sublists.append(numToOneHot(item, 13))

            if not isActive:
                sublists.append(np.zeros(activeSize))
            else:
                active = player['active'][monId]
                newlySwitched = bool(active['newlySwitched'])
                sublists.append(numToOneHot(newlySwitched, 2))

                ability = enumId('active-ability', active['ability'])
                sublists.append(numToOneHot(ability, 24))

                addedType = enumId('active-addedType', active['addedType'])
                sublists.append(numToOneHot(addedType, 13))

                moves = active['moves']
                moveList = [numToOneHot(m, 3) for m in moves]
                sublists.append(np.concatenate(moveList))

                boosts = active['boosts']
                boostList = [numToOneHot(b, 13) for b in boosts]
                sublists.append(np.concatenate(boostList))

                vols = active['volatiles']
                volList = np.zeros(10*10)
                for id in vols:
                    n = enumId('active-volatiles', id)
                    duration = numToOneHot(vols[id], 10)
                    insertSublist(volList, n, duration)
                sublists.append(volList)

            insertSublist(monList, monPlace, np.concatenate(sublists))

        playerList = np.concatenate([playerList, monList])

        #side conditions
        scList = np.zeros(10 * 10)
        for sideId in player['sideConditions']:
            n = enumId('player-sc', sideId)
            insertSublist(scList, n, numToOneHot(player['sideConditions'][sideId], 10))

        playerList = np.concatenate([playerList, scList])

        stateTensor = np.concatenate([stateTensor, playerList])

    return stateTensor

#turns the action into a tensor (really just an array)
#structure should be constant across runs
#as long as idMap stays the same
#always assumes singles or doubles
def actionToTensor(action):
    parts = [p.strip() for p in action.split(',')]
    if len(parts) == 1:
        parts.append('pass')

    actionTensor = np.zeros(0)
    #types of action (3) + max number of team combos in vgc (90)
    partSize = 3 + 90
    for p in parts:

        if p == 'pass':
            partList = np.zeros(partSize)

        elif 'switch' in p:
            target = p.split(' ')[1]
            targetNum = enumId('switch-target', target)
            partList = np.concatenate([numToOneHot(0, 3), numToOneHot(targetNum, 90)])

        elif 'team' in p:
            team = p.split(' ')[1]
            teamNum = enumId('team', team)
            partList = np.concatenate([numToOneHot(1, 3), numToOneHot(teamNum, 90)])

        elif 'move' in p:
            data = p.split(' ')
            move = data[1]
            moveNum = enumId('move-move', move)
            if len(data) < 3:
                targetNum = 0
            else:
                target = data[2]
                targetNum = enumId('move-target', target)
            partList = np.concatenate([numToOneHot(2, 3), numToOneHot(moveNum, 4), numToOneHot(moveNum, 86)])

        actionTensor = np.concatenate([actionTensor, partList])

    return actionTensor


def toInput(state, action1, action2):
    return np.concatenate([stateToTensor(state), actionToTensor(action1), actionToTensor(action2)])


#these are the nice parts of the action enumeration code

#action -> number
enumActionMap = None
#number -> action
denumActionMap = None

def enumAction(action):
    if not enumActionMap:
        genActionMap()
    #convert singles actions to a canonical form
    if ',' not in action and 'team' not in action:
        action += ',pass'
    #if there's a move with no target, set the target to 1
    fixedAction = []
    for part in action.split(','):
        part = part.strip()
        if 'move' in part:
            components = part.split(' ')
            #add a default target
            if len(components) == 2:
                components.append('1')
            part = ' '.join(components)
        fixedAction.append(part)

    action = ' ' + ','.join(fixedAction)
    return enumActionMap[action]

#do we ever even use denumAction()?

#this returns actions in canonical doubles form, not singles
#so if you're playing singles, cut off the ',pass' at the end
#def denumAction(n):
    #if not denumActionMap:
        #genActionMap()
    #return denumAction[n]

def genActionMap():
    global denumActionMap
    global enumActionMap
    denumActionMap = {}
    enumActionMap = {}

    #n is the number of unique actions
    #i is the enumerated actions, which has a lot of duplicates
    n = 0
    #minimum number to reach all actions
    for i in range(43200):
        action = _denumAction(i)
        if not action in enumActionMap:
            denumActionMap[n] = action
            enumActionMap[action] = n
            n += 1

#this is the ugly part

#converts a number to a doubles action (e.g. 'pass,pass', 'move 1 2, switch 2')
#this will assign all possible actions in some lower bound (<43200)
#this function is only used for the inital action enumeration generation
#so don't call this unless you're trying to make an enumeration
def _denumAction(n):
    actionType = n % 10
    n = n // 10
    #team
    if actionType == 9:
        #can pick 1-6 mons
        numPicked = (n % 6) + 1
        n = n // 6
        team = ['0' for _ in range(6)]
        for i in range(6, 0, -1):
            spot = n % i
            n = n // i
            #find the index of the spotth 0
            for j in range(6):
                if team[j] == '0':
                    spot -= 1
                if spot < 0:
                    team[j] = str(6 - i + 1)
                    break
        team = team[0:numPicked]
        return ' team ' + ''.join(team)

    actionType1 = actionType % 3
    actionType2 = actionType // 3

    actions = []
    for at in [actionType1, actionType2]:
        #move
        if at == 0:
            action = n % 16
            n = n // 16
            move = (action % 4) + 1
            target = (action // 4) - 2
            #0 isn't a valid target
            if target >= 0:
                target += 1
            actions.append('move ' + str(move) + ' ' + str(target))
        #switch
        elif at == 1:
            target = (n % 6) + 1
            n = n // 6
            actions.append('switch ' + str(target))
        #pass
        elif at == 2:
            actions.append('pass')

    return ' ' + ','.join(actions)





#turns a number into a one-hot representation
#0-indexed
#takes booleans too
def numToOneHot(num, size):
    if type(num) == bool and num:
        num = 1
    elif type(num) == bool and not num:
        num = 0
    xs = np.zeros(size)
    xs[num] = 1
    return xs

#copies the one-hot into the list at the given position
#defaults to sizes being in multiples of the one-hot
def insertSublist(xs, pos, oneHot, size=None):
    if size == None:
        size = len(oneHot)
    np.put(xs, range(pos * size, (pos+1) * size), oneHot)

if __name__ == '__main__':
    actions = {}
    for i in range(43200):
        action = denumAction(i)
        if action not in actions:
            print(action)
            actions[action] = True
