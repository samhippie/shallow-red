#!/usr/bin/env python3

import subprocess
import asyncio
import random
import collections
import math


#1v1 teams in packed format
teams = [
    '|mimikyu|mimikiumz||willowisp,playrough,swordsdance,shadowsneak|Jolly|240,128,96,,,44|||||]|zygardecomplete|groundiumz||thousandarrows,coil,substitute,rockslide|Impish|248,12,248,,,|||||]|volcarona|buginiumz|H|bugbuzz,quiverdance,substitute,overheat|Timid|,,52,224,,232||,0,,,,|||',

    '|naganadel|choicespecs||sludgewave,dracometeor,hiddenpowergrass,fireblast|Timid|56,,,188,64,200||,0,,,,|||]|zygarde|groundiumz|H|coil,substitute,bulldoze,thousandarrows|Impish|252,,220,,,36|||||]|magearna|fairiumz||calmmind,painsplit,irondefense,fleurcannon|Modest|224,,160,,,124||,0,,,,|||'
]

#location of the modified ps executable
PS_PATH = '/home/sam/builds/Pokemon-Showdown/pokemon-showdown'
PS_ARG = 'simulate-battle'

#hard coded moveset
#not all of these are legal, but we can deal with that later
moveSet = []
#no switching in 1v1
for i in range(4):
    for extra in ['', ' mega', 'zmove']:
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
    #REQUEST has format (REQUEST, state, message)
    REQUEST = 1
    #ERROR always refers to the previous message, has format (ERROR,)
    ERROR = 2
    #END has format (END, reward), reward is 1 for win, -1 for loss
    END = 3

    def __init__(self, ps):
        self.ps = ps
        self.state = 0
        self.cmdQueue = asyncio.Queue()
        self.p1Queue = asyncio.Queue()
        self.p2Queue = asyncio.Queue()

    #starts game in background
    async def startGame(self):
        asyncio.ensure_future(self.playGame())

    #handles init, starts input loop, handles output
    async def playGame(self):
        #commands to get the battle going
        initCommands = [
            '>start {"formatid":"gen71v1"}',
            '>player p1 {"name":"bot1", "team":"' + teams[0] + '"}',
            '>player p2 {"name":"bot2", "team":"' + teams[1] + '"}',
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
                running = False
            self.ps.stdin.write(bytes(cmd + '\n', 'UTF-8'))

    #reads input and queues player requests
    async def inputLoop(self):
        curQueue = None
        running = True
        while running:
            line = (await self.ps.stdout.readline()).decode('UTF-8')

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
                await curQueue.put((Game.REQUEST, self.state, message))

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

#generates prob table, which can be used as a policy for playing
#ps must not have a game running
#(this will change later when we play past team preview)
#has 2 prob tables as we have 2 separate agents
async def montecarloSearch(ps, limit=100,
        probTable1=collections.defaultdict(lambda: (0,0)),
        probTable2=collections.defaultdict(lambda: (0,0))):
    for i in range(limit):
        game = Game(ps)
        await game.startGame()
        #TODO
        #this is where we will try to match game's state to a given state
        await asyncio.gather(
                montecarloPlayerImpl(game.p1Queue, game.cmdQueue,
                    ">p1", probTable1),
                montecarloPlayerImpl(game.p2Queue, game.cmdQueue,
                    ">p2", probTable2))
    return (probTable1, probTable2)

#runs through a monte carlo playout for a single player
#so two of these need to be running for a 2 player game
#probTable maps (state, action) to (win, count)
#uct_c is the c constant used in the UCT calculation
async def montecarloPlayerImpl(requestQueue, cmdQueue, cmdHeader, probTable, uct_c=1.414):
    #types of request
    TEAM = 1
    TURN = 2
    SWITCH = 3

    #need to track these so we can correct errors
    prevState = None
    prevRequestType = None
    prevAction = None

    #history so we can update probTable
    history = []

    running = True
    randomPlayout = False
    while running:
        request = await requestQueue.get()

        if request[0] == Game.REQUEST or request[0] == Game.ERROR:
            if request[0] == Game.ERROR:
                state = prevState
                requestType = prevRequestType
                #punish the action that led to an error with 100 losses
                key = (state, requestType, prevAction)
                win, count = probTable[key]
                probTable[key] = win, (count + 100)
                #scrub the last action from history
                history = history[0:-1]
            else:
                state = request[1]
                message = request[2]
                if message[0:14] == '{"teamPreview"':
                    requestType = TEAM
                else:
                    requestType = TURN

            if requestType == TEAM:
                actions = teamSet
            elif requestType == TURN:
                actions = moveSet# + switchSet

            if randomPlayout:
                bestAction = random.choice(actions)
            else:
                #use upper confidence bound to pick the action
                #see the wikipedia MCTS page for details
                uctVals = []
                total = 0
                bestAction = None

                #need to get the total visit count first
                for action in actions:
                    key = (state, requestType, action)
                    win, count = probTable[key]
                    total += count

                #now we find the best UCT
                bestUct = None
                for action in actions:
                    key = (state, requestType, action)
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
            history.append((state, requestType, bestAction))
            prevAction = bestAction
            prevState = state
            prevRequestType = requestType
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

            running = False

async def main():
    ps = await asyncio.create_subprocess_exec(PS_PATH, PS_ARG,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE)

    probTable1, probTable2 = await montecarloSearch(ps, limit=100)
    print('prob table 1')
    for key in probTable1:
        result = probTable1[key]
        if(result[1] > 0 and result[1] < 100):
            print(key, '=>', probTable1[key], 100 * result[0] / result[1])
    print('prob table 2')
    for key in probTable2:
        result = probTable2[key]
        #<=0 means the action was never actually picked
        #>=100 means the action was illegal
        if(result[1] > 0 and result[1] < 100):
            print(key, '=>', probTable2[key], 100 * result[0] / result[1])

    ps.terminate()


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
