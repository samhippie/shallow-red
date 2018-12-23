#!/usr/bin/env python3

import asyncio
import json
import numpy as np
import random
import sys

uselessPrefixes = [
    'player', 'teamsize', 'gametype', 'gen', 'tier', 'seed',
    'rule', 'c', 'clearpoke\n', 'teampreview', 'start',
]
#turns a line into a series of tokens that can be added to an infoset
#takes the player to normalize between p1 and p2
def tokenize(line, player):
    if not line.startswith('|'):
        return []

    #split by '|', ':', ',', and '/'
    line = line.replace(':', '|').replace(',', '|').replace('/', '|')
    tokens = line.split('|')

    #filter out useless lines
    if tokens[1] in uselessPrefixes:
        return []

    #make it so all players see things from p1's perspective
    if player == 1:
        tokens = [token.replace('p2', 'p3').replace('p1', 'p2').replace('p3', 'p1') for token in tokens]

    #TODO remove any nicknames

    tokens.append('|')

    return [token.strip() for token in tokens[1:]]

#returns a seed that can be converted directly to a string and sent to PS
def getSeed():
    return [
        random.random() * 0x10000,
        random.random() * 0x10000,
        random.random() * 0x10000,
        random.random() * 0x10000,
    ]
        
#handles input and output of a single match
#players should read requests from p1Queue/p2Queue
#players should send responses to cmdQueue
#start by calling playGame(), which starts
#the cmdQueue consumer and p1Queue/p2Queue producers
#when the game is over, send 'reset' to cmdQueue
class Game:

    #new constructor for the more generic game implementation
    #history is now a set of (seed, player, action) tuples
    def __init__(self, ps, format, seed=None, names=['bot1', 'bot2'], history=[], verbose=False, file=sys.stdout):
        self.ps = ps
        self.format = format
        self.seed = seed
        self.names = names
        self.verbose = verbose
        self.file = file

        self.history = history

        self.waitingOnAction = False

        if format == 'singles':
            self.psFormat = 'anythinggoes'
        elif format == '2v2':
            self.psFormat = '2v2doubles'
        elif format == 'vgc':
            self.psFormat = 'vgc2019sunseries'
        else:
            self.psFormat = format

        self.format = format


        #request queues
        self.reqQueues = [asyncio.Queue(), asyncio.Queue()]
        for i in range(2):
            self.reqQueues[i].put_nowait({'teambuilding': True})

        #infosets are stored as a list of strings, which are basically tokens
        #individual events should have an end token at the end to separate them
        self.infosets = [['start', '|'], ['start', '|']]

        loop = asyncio.get_event_loop()
        self.winner = loop.create_future()

    async def startGame(self):
        await self.sendCmd('>start {"formatid":"gen7' + self.psFormat + '"' + (',"seed":' + str(self.seed) if self.seed else '') + '}')

        for (seed, player, action) in self.history:
            if seed != None:
                await self.sendCmd('>resetPRNG ' + seed)
            await self.sendCmd('>p' + str(player+1) + ' ' + action)

        asyncio.ensure_future(self.runInputLoop())


    #puts the requests in their queues
    #and fills in the infosets
    #and determines when the game is over
    async def runInputLoop(self):

        async def getLine():
            return (await self.ps.stdout.readline()).decode('UTF-8')

        while True:
            line = await getLine()

            if self.verbose:
                print(line, end='', file=self.file)

            if line == '|split\n':
                #private info
                #p1 infoset
                self.infosets[0] += tokenize(await getLine(), 0)
                #p2 infoset
                self.infosets[1] += tokenize(await getLine(), 1)
                #spectator view, throw out 
                await getLine()
                #omniscient view, print if verbose
                outLine = await getLine()
                if self.verbose:
                    print(outLine, end='', file=self.file)

            #p1 or p2 is about to get a request
            elif line == 'p1\n':
                curQueue = self.reqQueues[0]
            elif line == 'p2\n':
                curQueue = self.reqQueues[1]

            elif line.startswith('|request'):
                message = line[9:]
                message = json.loads(message)
                await curQueue.put(message)

            elif line.startswith('|error'):
                print('ERROR', line, file=sys.stderr)
                #this should never happen

            elif line.startswith('|win'):
                winner = line[5:-1]
                self.winner.set_result(winner)
                if winner == 'bot1':
                    winPlayer = 0
                    losePlayer = 1
                else:
                    winPlayer = 1
                    losePlayer = 0

                await self.reqQueues[winPlayer].put({'win': True})
                await self.reqQueues[losePlayer].put({'win': False})

                break
            else:
                #public info
                self.infosets[0] += tokenize(line, 0)
                self.infosets[1] += tokenize(line, 1)



    #returns whose turn it is
    async def getPlayer(self):
        if len(self.p1Queue) > 0:
            return 0
        else:
            return 1

    #gets the player and actions for the player
    #stores values so safe to call multiple times
    #values will be refreshed if a players has taken a turn
    #takeAction() expects that the given action is for the last turn returned by getTurn()
    async def getTurn(self):
        if not self.waitingOnAction:
            if self.reqQueues[0].qsize() > 0:
                self.curReq = await self.reqQueues[0].get()
                self.curPlayer = 0
            else:
                self.curReq = await self.reqQueues[1].get()
                self.curPlayer = 1
            self.curActions = getMoves(self.format, self.curReq)
        return (self.curPlayer, self.curReq, self.curActions)

    #gets the infoset (i.e. visible history i.e. state) for the given player
    def getInfoset(self, player):
        return self.infosets[player]

    async def takeAction(self, player, req, action):
        self.waitingOnAction = False
        self.infosets[player].append(action)
        if 'teambuilding' in req:
            cmd = '>player p' + str(player+1) + ' {"name":"' + self.names[player] + '", "avatar": "43", "team":"' + action + '"}'
            await self.sendCmd(cmd)
        elif action == 'nop':
            pass
        else:
            await self.sendCmd(action, player)
        

    async def sendCmd(self, cmd, player=None):
        if player != None:
            header = '>p' + str(player+1) + ' '
        else:
            header = ''
        if(self.verbose):
            print(header + cmd, file=self.file)
        self.ps.stdin.write(bytes(header + cmd + '\n', 'UTF-8'))

    async def resetSeed(self):
        seed = getSeed()
        await self.sendCmd('>resetPRNG ' + seed)
        return seed



#the code for determining moves
#a bit messy because it was originally in its own file
#it might be worth reorganizing this (again) and making each game have its own folder


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
    #not caching because req now includes things like seeds
    return getMovesImpl(format, req)
    #key = (format, str(req))
    #if not key in actionCache:
        #actionCache[key] = getMovesImpl(format, req)
    #return actionCache[key]

#this takes the req as a dict
#this works for anything that doesn't require switching
def getMovesImpl(format, req):
    if 'wait' in req:
        return [' noop']
    elif 'teambuilding' in req:
        #for now, we'll just give a pool of teams to pick from
        teams = [
            '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|M||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,|M|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,freezedry|Quiet|252,4,,252,,|M||||',
            '|charmander|leftovers||flamethrower,icebeam,dragondance,hyperbeam|Modest|,,,252,4,252|M||||]|bulbasaur|lifeorb||gigadrain,powerwhip,sludgebomb,rockslide|Adamant|252,252,,,,4|M||||]|squirtle|lifeorb||fakeout,earthquake,hydropump,freezedry|Timid|,4,,252,,252|M||||',
        ]
        return teams
    elif 'win' in req:
        return []
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
                        targets = []
                    #elif move['target'] == 'allySide':
                        #targets = ['-1' if i == 1 else '-2']
                    elif move['target'] in ['all', 'self', 'allAdjacentFoes', 'allAdjacent', 'randomNormal', 'foeSide', 'allySide']:
                        targets = ['']
                    elif move['target'] in ['normal', 'any']:
                        targets = ['-1' if i == 1 else '-2', '1', '2']
                    if len(targets) > 0:
                        for target in targets:
                            actions.append('move ' + str(j+1) + ' ' + target)
                    else:
                        actions.append('move ' + str(j+1))

            #pick the possible switching targets
            #TODO check how this works with shadow tag etc
            if not 'trapped' in req['active'][i]:
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


def prettyPrintMove(jointAction, req):
    action = jointAction.split(',')
    actionText = []
    for k in range(len(action)):
        a = action[k]
        a = a.strip()
        if 'pass' in a:
            actionText.append('pass')
        elif 'move' in a:
            parts = a.split(' ')
            moveNum = int(parts[1])
            if len(parts) < 3:
                targetNum = 0
            else:
                targetNum = int(parts[2])
            move = req['active'][k]['moves'][moveNum-1]['move']
            if targetNum != 0:
                actionText.append(move + ' into slot ' + str(targetNum))
            else:
                actionText.append(move)
        elif 'team' in a:
            actionText.append(a)
        elif 'switch' in a:
            targetNum = int(a.strip().split(' ')[1])
            mon = req['side']['pokemon'][targetNum-1]
            actionText.append('switch to ' + mon['details'])
        elif 'noop' in a:
            actionText.append('wait')
        else:
            actionText.append('unknown action: ' + a)
    actionString = ','.join(actionText)

    return actionString
