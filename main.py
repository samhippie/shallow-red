#!/usr/bin/env python3

import sys
import subprocess
import asyncio
import random
import collections
import math
import numpy as np
import copy


#1v1 teams in packed format
testTeams = [
    '|mimikyu|mimikiumz||willowisp,playrough,swordsdance,shadowsneak|Jolly|240,128,96,,,44|||||]|zygardecomplete|groundiumz||thousandarrows,coil,substitute,rockslide|Impish|248,12,248,,,|||||]|volcarona|buginiumz|H|bugbuzz,quiverdance,substitute,overheat|Timid|,,52,224,,232||,0,,,,|||',

    '|naganadel|choicespecs||sludgewave,dracometeor,hiddenpowergrass,fireblast|Timid|56,,,188,64,200||,0,,,,|||]|zygarde|groundiumz|H|coil,substitute,bulldoze,thousandarrows|Impish|252,,220,,,36|||||]|magearna|fairiumz||calmmind,painsplit,irondefense,fleurcannon|Modest|224,,160,,,124||,0,,,,|||',

    '|pyukumuku|psychiumz|H|lightscreen,recover,soak,toxic|Sassy|252,,4,,252,||,0,,,,0|||]|charizardmegax|charizarditex||flamecharge,outrage,flareblitz,swordsdance|Jolly|64,152,40,,,252|||||]|mew|keeberry||taunt,willowisp,roost,amnesia|Timid|252,,36,,,220||,0,,,,|||',

    '|tapulele|psychiumz||psychic,calmmind,reflect,moonblast|Calm|252,,60,,196,||,0,,,,|||]|charizard|charizarditex||willowisp,flamecharge,flareblitz,outrage|Jolly|252,,,,160,96|||||]|pheromosa|fightiniumz||bugbuzz,icebeam,focusblast,lunge|Modest|,,160,188,,160|||||',
]

#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

#hard coded moveset
#not all of these are legal, but we can deal with that later
moveSet = []
#no switching in 1v1
for i in range(4):
    for extra in ['', ' mega', ' zmove']:
        moveSet.append(' move ' + str(i+1) + extra)

#hard coded team preview combinations
#for 1v1, we only need to specify 1 mon out of 3
teamSet = []
for i in range(3):
    teamSet.append(' team ' + str(i+1))

#handles input and output of a single match
#players should read requests from p1Queue/p2Queue
#players should send responses to cmdQueue
#start by calling playGame(), which starts
#the cmdQueue consumer and p1Queue/p2Queue producers
#when the game is over, send 'reset' to cmdQueue
class Game:

    #types of messages to player
    #REQUEST has format (REQUEST, state, request type)
    REQUEST = 1
    #ERROR always refers to the previous message, has format (ERROR,)
    ERROR = 2
    #END has format (END, reward), reward is 1 for win, -1 for loss
    END = 3

    #request types
    REQUEST_TEAM = 1
    REQUEST_TURN = 2
    REQUEST_SWITCH = 3


    def __init__(self, ps, teams, seed=None, verbose=False):
        #the pokemon showdown process
        self.ps = ps
        #a list of the two teams
        self.teams = teams
        #the hash of the current game state
        self.state = 0
        #send commands here to be sent to the process
        self.cmdQueue = asyncio.Queue()
        #read from these to get requests for actions
        self.p1Queue = asyncio.Queue()
        self.p2Queue = asyncio.Queue()
        self.seed = seed
        #will have a string value of the winner when the game is over
        loop = asyncio.get_event_loop()
        self.winner = loop.create_future()
        #whether to print out PS's output
        self.verbose=verbose
        #the task of reading PS's output, which might need to be cancelled
        #if the game ends early
        self.inputTask = None

    #starts game in background
    async def startGame(self):
        asyncio.ensure_future(self.playGame())

    #handles init, starts input loop, handles output
    async def playGame(self):
        #commands to get the battle going
        initCommands = [
            '>start {"formatid":"gen71v1"' + (',"seed":' + str(self.seed) if self.seed else '') + '}',
            '>player p1 {"name":"bot1", "team":"' + self.teams[0] + '"}',
            '>player p2 {"name":"bot2", "team":"' + self.teams[1] + '"}',
        ]

        for cmd in initCommands:
            await self.cmdQueue.put(cmd)

        #start up the player input
        asyncio.ensure_future(self.inputLoop())

        #write output in a nice asynchronous way
        running = True
        while running:
            cmd = await self.cmdQueue.get()
            if cmd == 'reset':
                self.inputTask.cancel()
                running = False
            else:
                self.ps.stdin.write(bytes(cmd + '\n', 'UTF-8'))

    #reads input and queues player requests
    async def inputLoop(self):
        curQueue = None
        running = True
        skipLine = False
        loop = asyncio.get_event_loop()
        while running:
            self.inputTask = loop.create_task(self.ps.stdout.readline())
            try:
                line = (await self.inputTask).decode('UTF-8')
            except:
                #input got cancelled
                break

            #we skip lines that are duplicated and mess up the replay
            if line == '|split\n':
                #skips the |split line
                #the next line will be repeated 4 times, only show
                #the last, which is more detailed
                skipLine = 4

            if self.verbose and skipLine <= 0:
                print(line, end='')

            if skipLine > 0:
                skipLine -= 1

            #sets the recipient for the next request
            if line == 'p1\n':
                curQueue = self.p1Queue
            elif line == 'p2\n':
                curQueue = self.p2Queue

            #store the current game state
            elif line.startswith('|c|~Zarel\'s Mom|'):
                self.state = int(line[16:])

            #sends the request to the current recipient
            elif line.startswith('|request'):
                message = line[9:]
                requestType = self.getRequestType(message)
                await curQueue.put((Game.REQUEST, (self.state, requestType)))

            #tells the current recipient their last response was rejected
            elif line.startswith('|error'):
                await curQueue.put((Game.ERROR,))

            #game is over, send out messages and clean up
            elif line.startswith('|win'):
                winner = line[5:-1]

                if winner == 'bot1':
                    await self.p1Queue.put((Game.END, 1))
                    await self.p2Queue.put((Game.END, -1))
                else:
                    await self.p1Queue.put((Game.END, -1))
                    await self.p2Queue.put((Game.END, 1))

                await self.cmdQueue.put('reset')
                running = False
                self.winner.set_result(winner)

            #a tie, which usually happens when the game ended early with >forcetie
            #can't just look for '|tie' as '|tier' is a common message
            elif line.startswith('|tie\n'):
                winner = 'the cat'

                await self.p1Queue.put((Game.END, 0))
                await self.p2Queue.put((Game.END, 0))

                await self.cmdQueue.put('reset')
                running = False
                self.winner.set_result(winner)

    def getRequestType(self, message):
        if message[0:14] == '{"teamPreview"':
            return Game.REQUEST_TEAM
        else:
            return Game.REQUEST_TURN
        #need to add switch and nothing


#generates prob table, which can be used as a policy for playing
#ps must not have a game running
#the start states and request types are used to set the game state
#we will try to get to the proper state
#(this will change later when we play past team preview)
#has 2 prob tables as we have 2 separate agents

#returns None if it failed to achieve the start state
#otherwise returns two prob tables
async def montecarloSearch(ps, teams, limit=100,
        seed=None, p1InitActions=[], p2InitActions=[],
        probTable1=collections.defaultdict(lambda: (0,0)),
        probTable2=collections.defaultdict(lambda: (0,0))):
    print(end='', file=sys.stderr)
    for i in range(limit):
        print('\rTurn Progress: ' + str(i) + '/' + str(limit), end='', file=sys.stderr)
        game = Game(ps, teams, seed=seed, verbose=False)
        await game.startGame()
        await asyncio.gather(
                montecarloPlayerImpl(game.p1Queue, game.cmdQueue,
                    ">p1", probTable1, errorPunishment=2*limit,
                    initActions=p1InitActions),
                montecarloPlayerImpl(game.p2Queue, game.cmdQueue,
                    ">p2", probTable2, errorPunishment=2*limit,
                    initActions=p2InitActions))
    print(file=sys.stderr)
    return (probTable1, probTable2)

#runs through a monte carlo playout for a single player
#so two of these need to be running for a 2 player game
#probTable maps (state, action) to (win, count)
#uct_c is the c constant used in the UCT calculation
#errorPunishment is how many losses an error counts as
#initActions is a list of initial actions that will be blindy taken
async def montecarloPlayerImpl(requestQueue, cmdQueue, cmdHeader, probTable,
        uct_c=1.414, errorPunishment=100, initActions=[]):
    #need to track these so we can correct errors
    prevState = None
    prevRequestType = None
    prevAction = None

    #history so we can update probTable
    history = []

    #we're going to be popping off this
    initActions = copy.deepcopy(initActions)

    running = True
    randomPlayout = False
    inInitActions = True
    while running:
        request = await requestQueue.get()

        if request[0] == Game.REQUEST or request[0] == Game.ERROR:
            if request[0] == Game.ERROR:
                if len(initActions) > 0:
                    print('WARNING got an error following initActions', file=sys.stderr)
                state = prevState
                #punish the action that led to an error with a bunch of losses
                key = (prevState, prevAction)
                win, count = probTable[key]
                probTable[key] = win, (count + errorPunishment)
                #scrub the last action from history
                history = history[0:-1]
            else:
                state = request[1]

            if state[1] == Game.REQUEST_TEAM:
                actions = teamSet
            elif state[1] == Game.REQUEST_TURN:
                actions = moveSet# + switchSet

            #check if we ran out of initActions on the previous turn
            #if so, we need to change the PRNG
            if inInitActions and len(initActions) == 0:
                inInitActions = False
                #no problem if both players reset the PRNG
                await cmdQueue.put('>resetPRNG')

            if len(initActions) > 0:
                bestAction = initActions[0]
                initActions = initActions[1:]
            elif randomPlayout:
                bestAction = random.choice(actions)
            else:
                #use upper confidence bound to pick the action
                #see the wikipedia MCTS page for details
                uctVals = []
                total = 0
                bestAction = None

                #need to get the total visit count first
                for action in actions:
                    key = (state, action)
                    win, count = probTable[key]
                    total += count

                #now we find the best UCT
                bestUct = None
                for action in actions:
                    key = (state, action)
                    win, count = probTable[key]
                    #never visited -> infinite UCT
                    #also means we start the random playout
                    if count == 0:
                        bestAction = action
                        randomPlayout = True
                        break
                    uct = win / count + uct_c * math.sqrt(math.log(total) / count)
                    if bestUct == None or uct > bestUct:
                        bestUct = uct
                        bestAction = action

            #save our action
            history.append((state, bestAction))

            prevAction = bestAction
            prevState = state
            #send out the action
            await cmdQueue.put(cmdHeader + bestAction)

        elif request[0] == Game.END:
            #update probTable with our history + result
            reward = request[1]
            for key in history:
                win, count = probTable[key]
                if reward == -1:
                    probTable[key] = (win, count+1)
                elif reward == 1:
                    probTable[key] = (win+1, count+1)
                elif reward == 0: # we shouldn't be seeing any ties in the MCTS loop, as I don't think you can get ties without a timer
                    print('WARNING tie in MCTS', file=sys.stderr)
                    probTable[key] = (win+0.5, count+1)

            running = False

async def playTestGame(limit=100):
    try:
        mainPs = await getPSProcess()
        searchPs = await getPSProcess()

        seed = [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

        teams = (testTeams[2], testTeams[3])
        game = Game(mainPs, teams=teams, seed=seed, verbose=True)

        await game.startGame()

        probTable1 = collections.defaultdict(lambda: (0,0))
        probTable2 = collections.defaultdict(lambda: (0,0))


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
                await montecarloSearch(searchPs,
                        teams,
                        limit=limit,
                        seed=seed,
                        p1InitActions=p1Actions,
                        p2InitActions=p2Actions,
                        probTable1=probTable1,
                        probTable2=probTable2)

                #this assumes that both player1 and player2 get requests each turn
                #which I think is accurate, but most formats will give one player a waiting request
                #except for errors

                async def playTurn(queue, probTable, actionList, cmdHeader):
                    #figure out what kind of action we need
                    request = await queue.get()
                    if request[1][1] == Game.REQUEST_TEAM:
                        actions = teamSet
                    elif request[1][1] == Game.REQUEST_TURN:
                        actions = moveSet
                    #get the probability of each action winning
                    probs = [win / (max(count, 1)) for win,count in [probTable[(request[1], action)] for action in actions]]
                    probSum = np.sum(probs)
                    if probSum == 0:
                        print('|c|' + cmdHeader + '|Turn ' + str(i) + ' seems impossible to win')
                    for i in range(len(actions)):
                        if probs[i] > 0:
                            print('|c|' + cmdHeader + '|Turn ' + str(i) + ' action:', actions[i], 'prob:', probs[i])
                    #pick according to the probability (or should we be 100% greedy?)
                    action = np.random.choice(actions, p=probs / probSum)
                    actionList.append(action)
                    await game.cmdQueue.put(cmdHeader + action)

                await playTurn(game.p1Queue, probTable1, p1Actions, '>p1')
                await playTurn(game.p2Queue, probTable2, p2Actions, '>p2')

        #gameTask = play()
        #asyncio.ensure_future(gameTask)
        gameTask = asyncio.ensure_future(play())
        winner = await game.winner
        gameTask.cancel()
        print('winner:', winner, file=sys.stderr)

    finally:
        mainPs.terminate()
        searchPs.terminate()

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
                    actions = teamSet
                elif req[1][1] == Game.REQUEST_TURN:
                    actions = moveSet

                action = random.choice(actions)
                await game.cmdQueue.put(cmdHeader + action)

        await game.startGame()
        await asyncio.gather(randomAgent(game.p1Queue, '>p1'),
                randomAgent(game.p2Queue, '>p2'))

        winner = await game.winner
        print('winner:', winner)
    finally:
        ps.terminate()

async def getPSProcess():
    return await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

async def main():
    await playTestGame(limit=1000)
    #await playRandomGame()

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
