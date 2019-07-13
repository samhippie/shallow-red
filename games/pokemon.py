#!/usr/bin/env python3

import asyncio
import copy
import json
import math
import numpy as np
import random
import subprocess
import sys

import config

#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

class _Context:
    async def __aenter__(self):
        print('making ps process', file=sys.stderr)
        self.ps = await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)
        return self.ps

    async def __aexit__(self, *args):
        print('terminating ps process', file=sys.stderr)
        self.ps.terminate()

#our context is a pokemon showdown process
def getContext():
    return _Context()

#really determined by format, but I don't want too many outputs
#for doubles and switching formats, we'll probably have to break each mon up, and then break attacking and switching up
numActions = 10

#mons are joined with ']' in between
#send setteam mon1]mon2]mon3
legalMons = [
    #'Bisharp||darkiniumz||suckerpunch,taunt,ironhead,falseswipe|Adamant|4,252,,,,252|||||',
    #'Alakazam||alakazite||disable,hiddenpowerfighting,psychic,knockoff|Modest|4,,,252,,252||,30,,,,|||',
    #'Salazzle||focussash||fireblast,icebeam,sludgewave,disable|Modest|4,,,252,,252||,0,,,,|||',
    'Salazzle||focussash||imprison,icebeam,,|Modest|4,,,252,,252||,0,,,,|||',
    'Bisharp||focussash||imprison,flamethrower,,|Modest|4,,,252,,252||,0,,,,|||',
    'Alakazam||focussash||imprison,scald,,|Modest|4,,,252,,252||,0,,,,|||',
    #'Hawlucha||aguavberry|1|acrobatics,brickbreak,drainpunch,firepunch|Adamant|252,252,,,,4|||||',
    #'Ferrothorn||rockyhelmet||leechseed,gyroball,powerwhip,knockoff|Sassy|252,4,,,252,||,,,,,0|||',
    #'Dragonite||choiceband|H|dragonclaw,earthquake,extremespeed,aquatail|Jolly|4,252,,,,252|||||',
]

#incomplete
teamPickingFormats = [
    '1v1',
]

uselessPrefixes = [
    'player', 'teamsize', 'gametype', 'gen', 'tier', 'seed',
    'rule', 'c', 'clearpoke\n', 'teampreview', 'start', '-hint',
]

with open('games/dataPokemon/pokemonTokenMapper.json', 'r') as f:
    tokenMapper = json.load(f)

#turns a line into a series of tokens that can be added to an infoset
#takes the player to normalize between p1 and p2
def tokenize(line, player, gameLine=True):
    if gameLine and not line.startswith('|'):
        return []

    if not line.strip():
        return []

    #split by '|', ':', ',', and '/'
    line = line.replace(':', '|').replace(',', '|').replace('/', '|')
    tokens = line.split('|')

    #filter out useless lines
    if len(tokens) > 1 and tokens[1] in uselessPrefixes:
        return []

    #make it so all players see things from p1's perspective
    #and also make tokens lowercase and strip whitespace
    if player == 1:
        tokens = [token.lower().strip().replace('p2', 'p3').replace('p1', 'p2').replace('p3', 'p1') for token in tokens]
    else:
        tokens = [token.lower().strip() for token in tokens]

    #no empty strings
    tokens = [token for token in tokens if token]

    #may turn each token into a set of tokens
    tokens = [(tokenMapper[token] if token in tokenMapper else [token]) for token in tokens]
    #concatenate so we get a simple flat list
    tokens = [inner for outer in tokens for inner in outer]

    #TODO remove any nicknames

    tokens.append('|')

    return tokens

#returns a seed that can be converted directly to a string and sent to PS
#actually 3 seeds, battle, p1, and p2
def getSeed():
    return [[math.floor(random.random() * 0x10000) for i in range(4)] for j in range(3)]
        
#handles input and output of a single match
#players should read requests from p1Queue/p2Queue
#players should send responses to cmdQueue
#start by calling playGame(), which starts
#the cmdQueue consumer and p1Queue/p2Queue producers
#when the game is over, send 'reset' to cmdQueue
class Game:

    #new constructor for the more generic game implementation
    #history is now a set of (seed, action) tuples for each player
    def __init__(self, context, history=[[],[]], seed=None, saveTrajectories=False, names=['bot1', 'bot2'], verbose=False, file=sys.stdout):
        self.ps = context 
        self.format = config.Pokemon.format
        self.seed = seed
        self.names = names
        self.verbose = verbose
        self.file = file
        self.saveTrajectories = saveTrajectories

        self.history = history

        self.waitingOnAction = False

        if self.format == 'singles':
            self.psFormat = 'anythinggoes'
        elif self.format == '2v2':
            self.psFormat = '2v2doubles'
        elif self.format == 'vgc':
            self.psFormat = 'vgc2019sunseries'
        else:
            self.psFormat = self.format


        if self.saveTrajectories:
            #list of (infoset, action)
            #for each player
            self.prevTrajectories = [[],[]]


        #request queues
        self.reqQueues = [asyncio.Queue(), asyncio.Queue()]
        #we're now handling teambuilding separately
        #for i in range(2):
            #self.reqQueues[i].put_nowait({'teambuilding': True})

        #infosets are stored as a list of strings, which are basically tokens
        #individual events should have an end token at the end to separate them
        self.infosets = [['start', '|'], ['start', '|']]

        loop = asyncio.get_event_loop()
        self.winner = loop.create_future()

    async def startGame(self):
        await self.sendCmd('>start {"formatid":"gen7' + self.psFormat + '"' + (',"seed":' + str(self.seed[0]) if self.seed else '') + '}')

        #state for team builder
        #this will probably grow enough to be its own class
        self.inTeamPicker = True
        self.pickedMons = [[],[]]

        #mons that need to be mega evolved on the next command
        self.toMega = [[],[]]

        asyncio.ensure_future(self.runInputLoop())

        #have to execute each player's history
        #is it necessary to copy here?
        h = [copy.copy(self.history[0]), copy.copy(self.history[1])]
        while len(h[0]) or len(h[1]):
            #both players may have requests, but we might only have history for one
            if len(h[0]) and not len(h[1]):
                prefPlayer = 0
            elif len(h[1]) and not len(h[0]):
                prefPlayer = 1
            else:
                prefPlayer = None

            player, req, actions = await self.getTurn(prefPlayer)
            seed, actionIndex = h[player][0]
            del h[player][0]
            if seed != None:
                await self.sendCmd('>resetPRNG ' + seed[0])
            await self.takeAction(player, actionIndex)



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

            #old-style split
            #PS changed how they split
            #https://github.com/Zarel/Pokemon-Showdown/commit/7e4929a39f72a553604bdff58e37b5c95e695e04
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

            elif line.startswith('|split|'):
                #direct secret player line to player
                player = 0 if line[7:9] == 'p1' else 1
                self.infosets[player] += tokenize(await getLine(), player)
                #direct public line to other player and print out
                outLine = await getLine()
                self.infosets[(player + 1) % 2] += tokenize(outLine, (player + 1) % 2)
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

            elif line.startswith('|error|[Unavailable choice]'):
                #don't actually have to do anything
                pass

            elif line.startswith('|error'):
                print('ERROR', line, file=sys.stderr)
                print('looks like a real error', file=sys.stderr)
                quit()

            elif line.startswith('|win'):
                winner = line[5:-1]
                self.winner.set_result(winner)
                if winner == self.names[0]:
                    winPlayer = 0
                    losePlayer = 1
                else:
                    winPlayer = 1
                    losePlayer = 0

                await self.reqQueues[winPlayer].put({'win': 1})
                await self.reqQueues[losePlayer].put({'win': -1})
                break

            elif line == '|tie\n':#can't use startswith because of 'tier'
                winner = line[5:-1]
                self.winner.set_result(-1)#is setting this to -1 a good idea? let's see if anything breaks

                await self.reqQueues[0].put({'win': 0})
                await self.reqQueues[1].put({'win': 0})
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
    async def getTurn(self, prefPlayer=None):
        if not self.waitingOnAction:
            #optionally, we might get a request for a specific player's turn
            #even if prefPlayer is specified, we won't invalidate outstanding requests
            #so we still obey waitingOnAction
            if self.inTeamPicker and len(self.pickedMons[0]) < 3:
                self.curReq = {'teambuilding': self.pickedMons[0]}
                self.curPlayer = 0
            elif self.inTeamPicker and len(self.pickedMons[1]) < 3:
                self.curReq = {'teambuilding': self.pickedMons[1]}
                self.curPlayer = 1
            elif prefPlayer != None:
                self.curReq = await self.reqQueues[prefPlayer].get()
                self.curPlayer = prefPlayer
            elif self.reqQueues[0].qsize() > 0:
                self.curReq = await self.reqQueues[0].get()
                self.curPlayer = 0
            elif self.reqQueues[1].qsize() > 0:
                self.curReq = await self.reqQueues[1].get()
                self.curPlayer = 1
            else:
                #we need to loop like this in some rare instances, like when a move gets imprisoned
                while True:
                    try:
                        self.curReq = await asyncio.wait_for(self.reqQueues[0].get(), timeout=.3)
                        self.curPlayer = 0
                        break
                    except asyncio.TimeoutError:
                        try:
                            self.curReq = await asyncio.wait_for(self.reqQueues[1].get(), timeout=.3)
                            self.curPlayer = 1
                            break
                        except asyncio.TimeoutError:
                            pass

            self.curActionCmds = self.getMoves(self.curPlayer, self.curReq)
            self.curActions = [prettyPrintMove(a, self.curReq) for a in self.curActionCmds]
        return (self.curPlayer, self.curReq, self.curActions)

    #gets the infoset (i.e. visible history i.e. state) for the given player
    def getInfoset(self, player):
        if self.curPlayer == player:
            infoContext = ['OPTIONS']
            for i, action in enumerate(self.curActions):
                infoContext += ['@', str(i)] + action
            return self.infosets[player] + infoContext
        else:
            return self.infosets[player]

    async def takeAction(self, player, actionIndex):
        action = self.curActions[actionIndex]
        actionCmd = self.curActionCmds[actionIndex]
        self.waitingOnAction = False

        if self.saveTrajectories:
            self.prevTrajectories[player].append((copy.copy(self.getInfoset(player)), actionIndex, copy.copy(self.curActions)))

        self.infosets[player] += ['CHOICE'] +  action

        if self.inTeamPicker and self.format not in teamPickingFormats:
            self.inTeamPicker = False
            await self.sendCmd('setteam ', 0)
            await self.sendCmd('setteam ', 1)
        elif self.inTeamPicker and len(self.pickedMons[player]) < 3:
            #must be team preview, add to list of mons
            self.pickedMons[player].append(actionCmd)
            #if both players now have enough mons, send team selection
            if len(self.pickedMons[0]) == 3 and len(self.pickedMons[1]) == 3:
                self.inTeamPicker = False
                await self.sendCmd('setteam ' + ']'.join(self.pickedMons[0]), 0)
                await self.sendCmd('setteam ' + ']'.join(self.pickedMons[1]), 1)
        elif '!makemega' in actionCmd:
            #just save to internal mega state
            mon = int(actionCmd.split(' ')[1])
            self.toMega[player].append(mon)
            #we've already marked a pokemon as going to mega, so we'll remove the option from the request
            req = self.curReq
            if len(self.toMega[player]) > 0 and 'active' in req:
                for i in range(len(req['active'])):
                    if 'canMegaEvo' in req['active'][i] and req['active'][i]['canMegaEvo'] and i in self.toMega[player]:
                        del req['active'][i]['canMegaEvo']

            #and we'll just reactivate the pending request
            self.curActionCmds = self.getMoves(self.curPlayer, self.curReq)
            self.curActions = [prettyPrintMove(a, self.curReq) for a in self.curActionCmds]
            self.waitingOnAction = True
        else:
            if len(self.toMega[player]) > 0:
                cmdlets = actionCmd.split(',')
                for i, mon in enumerate(self.toMega[player]):
                    cmdlets[i] += ' mega'
                actionCmd = ','.join(cmdlets)
                self.toMega[player] = []
            
            await self.sendCmd(actionCmd, player)
        

    async def sendCmd(self, cmd, player=None):
        if cmd == '!noop':
            #showdown doesn't expect any input
            return
        if cmd.startswith('setteam '):
            #teambuilding is handled outside of the actual game, so it's not a normal command
            team = cmd[8:]
            msg = '>player p' + str(player+1) + ' {"name":"' + self.names[player] + '", "avatar": "43"' + ((',"team":"' + team + '"') if team != '' else '') + ((',"seed":' + str(self.seed[player+1])) if self.seed else '') + '}'
            if(self.verbose):
                print('sending', msg, file=self.file)
            self.ps.stdin.write(bytes(msg + '\n', 'UTF-8'))
            return
        if player != None:
            header = '>p' + str(player+1) + ' '
        else:
            header = ''
        if(self.verbose):
            print('sending', header + cmd, file=self.file)
        self.ps.stdin.write(bytes(header + cmd + '\n', 'UTF-8'))

    #I'm upgrading the local version of PS, and I don't feel like porting this over
    #we haven't been using it, so it should be fine
    #async def resetSeed(self):
        #seed = getSeed()
        #await self.sendCmd('>resetPRNG ' + seed)
        #return seed

    def getMoves(self, player, req):
        
        return getMovesImpl(self.format, req)



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
    return ['team ' + ''.join([str(t) for t in team]) for team in teams]

#maps format => teams
#teamCache = {}

#maps (format, str(req)) => actions
actionCache = {}

doublesFormats = ['doubles', '2v2doubles', '2v2', 'vgc']

#this takes the req as a dict
#this works for anything that doesn't require switching
def getMovesImpl(format, req):
    if 'wait' in req:
        return ['!noop']
    elif 'teambuilding' in req:
        if format in teamPickingFormats:
            curMons = req['teambuilding']#this organization sucks
            availableMons = [m for m in legalMons if not m in curMons]
            return availableMons
        else:
            return ['']
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
                actions.append(','.join(set))
            elif len(actionCross) == 1:#only one mon left
                #need to pass some of the switches
                #the proper way is to replace all duplicates with a pass
                #I'm just going to hard code this for doubles
                actions.append(set[0] + ',pass')
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
                if 'canMegaEvo' in req['active'][i] and req['active'][i]['canMegaEvo']:
                    actions.append('!makemega ' + str(i))
                moves = req['active'][i]['moves']
                numNormalMoves = len(moves)
                if 'canZMove' in req['active'][i]:
                    moves += req['active'][i]['canZMove']
                for j, move in enumerate(moves):
                    if move is None:
                        continue
                    if ('disabled' in move and move['disabled']) or ('pp' in move and move['pp'] == 0):
                        continue
                    if format not in doublesFormats or 'target' not in move:
                        targets = []
                    #elif move['target'] == 'allySide':
                        #targets = ['-1' if i == 1 else '-2']
                    elif move['target'] in ['all', 'self', 'allAdjacentFoes', 'allAdjacent', 'randomNormal', 'foeSide', 'allySide']:
                        targets = []
                    elif move['target'] in ['normal', 'any']:
                        targets = ['-1' if i == 1 else '-2', '1', '2']
                    if len(targets) > 0:
                        for target in targets:
                            if j >= numNormalMoves:
                                actions.append('move ' + str(j+1 - numNormalMoves) + ' ' + target + ' zmove')
                            else:
                                actions.append('move ' + str(j+1) + ' ' + target)
                    else:
                        if j >= numNormalMoves:
                            actions.append('move ' + str(j+1 - numNormalMoves) + ' zmove')
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
                actions.append(','.join(set))
        return actions

def prettyPrintMove(jointAction, req):
    #player = 0 if req['side']['id'] == 'p1' else 1
    player = 0#actions shouldn't be player-specific
    return tokenize(prettyPrintMoveImpl(jointAction, req), player, gameLine=False)

def prettyPrintMoveImpl(jointAction, req):
    if 'teambuilding' in req:
        return jointAction
    action = jointAction.split(',')
    actionText = []
    for k in range(len(action)):
        a = action[k]
        a = a.strip()
        if a.startswith('pass'):
        #if 'pass' in a:
            actionText.append('pass')
        elif a.startswith('!makemega'):
            mon = a.split(' ')[1]
            actionText.append('mega evolve,' + mon)
        elif a.startswith('move'):
            parts = a.split(' ')
            zMove = parts[-1] == 'zmove'
            if zMove:
                del parts[-1]
            moveNum = int(parts[1])
            if len(parts) < 3:
                targetNum = 0
            else:
                targetNum = int(parts[2])
            move = req['active'][k]['moves'][moveNum-1]['move']
            if targetNum != 0:
                actionText.append(('Z|' if zMove else '') + move + ', into slot, ' + str(targetNum))
            else:
                actionText.append(('Z|' if zMove else '') + move)
        elif a.startswith('team'):
            team = a.split(' ')[1]
            monSummaries = []
            for j, monNum in enumerate(team):
                i = int(monNum)-1
                mon = req['side']['pokemon'][i]
                summary = mon['details'] + ',moves,' + (','.join(mon['moves']))
                monSummaries.append('team ' + str(j) + ',' + summary)
            teamAction = ','.join(monSummaries)
            actionText.append(teamAction)
        elif a.startswith('switch'):
            targetNum = int(a.strip().split(' ')[1])
            mon = req['side']['pokemon'][targetNum-1]
            actionText.append('switch to ' + mon['details'])
        elif a.startswith('!noop'):
            actionText.append('wait')
        else:
            actionText.append('unknown action: ' + a)
    actionString = ','.join(actionText)

    return actionString


if __name__ == '__main__':
    #test code for adding megas and z moves
    req = json.loads("""
            {"teamPreview":true,"maxTeamSize":1,"side":{"name":"bot2","id":"p2","pokemon":[{"ident":"p2: Voltorb","details":"Voltorb, L85","condition":"178/178","active":true,"stats":{"atk":77,"def":141,"spa":151,"spd":106,"spe":196},"moves":["doubleteam","lightscreen","headbutt","torment"],"baseAbility":"soundproof","item":"loveball","pokeball":"pokeball","ability":"soundproof"},{"ident":"p2: Magneton","details":"Magneton, L76","condition":"211/211","active":false,"stats":{"atk":114,"def":166,"spa":209,"spd":131,"spe":158},"moves":["toxic","shockwave","discharge","metalsound"],"baseAbility":"analytic","item":"flyiniumz","pokeball":"pokeball","ability":"analytic"},{"ident":"p2: Vaporeon","details":"Vaporeon, L72, M","condition":"286/286","active":false,"stats":{"atk":120,"def":134,"spa":193,"spd":174,"spe":105},"moves":["sleeptalk","laserfocus","aquatail","confide"],"baseAbility":"hydration","item":"glalitite","pokeball":"pokeball","ability":"hydration"},{"ident":"p2: Doduo","details":"Doduo, L88, F","condition":"169/169","active":false,"stats":{"atk":174,"def":137,"spa":94,"spd":107,"spe":185},"moves":["endure","aircutter","mimic","toxic"],"baseAbility":"runaway","item":"berryjuice","pokeball":"pokeball","ability":"runaway"},{"ident":"p2: Weepinbell","details":"Weepinbell, L82, M","condition":"201/201","active":false,"stats":{"atk":216,"def":112,"spa":178,"spd":110,"spe":112},"moves":["endure","morningsun","rest","synthesis"],"baseAbility":"gluttony","item":"rowapberry","pokeball":"pokeball","ability":"gluttony"},{"ident":"p2: Bellsprout","details":"Bellsprout, L89, M","condition":"193/193","active":false,"stats":{"atk":195,"def":69,"spa":220,"spd":83,"spe":83},"moves":["bulletseed","cut","round","growth"],"baseAbility":"chlorophyll","item":"fistplate","pokeball":"pokeball","ability":"chlorophyll"}]}}
            """)


    print(json.dumps(req, indent=2))
    moves = getMovesImpl('1v1', req)
    print(moves)
    print([prettyPrintMove(m, req) for m in moves])
    print(prettyPrintMove('team 12', req))
