#!/usr/bin/env python3

import asyncio
import json
import random
import sys

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
    REQUEST_WAIT = 4


    def __init__(self, ps, teams, format='1v1', seed=None, names=['bot1', 'bot2'], verbose=False, file=sys.stdout):
        #the pokemon showdown process
        self.ps = ps
        #a list of the two teams
        self.teams = teams
        self.format = format
        self.names = names
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
        #where we print to
        self.file = file
        #the task of reading PS's output, which might need to be cancelled
        #if the game ends early
        self.inputTask = None

    #starts game in background
    async def startGame(self):
        asyncio.ensure_future(self.playGame())

    #handles init, starts input loop, handles output
    async def playGame(self):
        #PS doesn't actually enforce banlists, so as
        #long as the format is close we should be fine
        if self.format == 'singles':
            psFormat = 'anythinggoes'
        elif self.format == '2v2':
            psFormat = '2v2doubles'
        elif self.format == 'vgc':
            psFormat = 'vgc2019sunseries'
        else:
            psFormat = self.format

        #commands to get the battle going
        initCommands = [
            '>start {"formatid":"gen7' + psFormat + '"' + (',"seed":' + str(self.seed) if self.seed else '') + '}',
            '>player p1 {"name":"' + self.names[0] + '", "avatar": "43", "team":"' + self.teams[0] + '"}',
            '>player p2 {"name":"' + self.names[1] + '", "avatar": "43", "team":"' + self.teams[1] + '"}',
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
            elif 'noop' in cmd:
                #don't need to send anything to do nothing
                pass
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
                print(line, end='', file=self.file)

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
                message = json.loads(message)
                await curQueue.put((Game.REQUEST, message))

            #tells the current recipient their last response was rejected
            #in the future errors shouldn't happen
            elif line.startswith('|error'):
                print('ERROR', line, file=sys.stderr)
                #await curQueue.put((Game.ERROR,))

            #game is over, send out messages and clean up
            elif line.startswith('|win'):
                winner = line[5:-1]

                self.winner.set_result(winner)

                if winner == 'bot1':
                    await self.p1Queue.put((Game.END, 1))
                    await self.p2Queue.put((Game.END, -1))
                else:
                    await self.p1Queue.put((Game.END, -1))
                    await self.p2Queue.put((Game.END, 1))

                await self.cmdQueue.put('reset')
                running = False

            #a tie, which usually happens when the game ended early with >forcetie
            #can't just look for '|tie' as '|tier' is a common message
            elif line.startswith('|tie\n'):
                winner = 'the cat'

                await self.p1Queue.put((Game.END, 0))
                await self.p2Queue.put((Game.END, 0))

                await self.cmdQueue.put('reset')
                running = False
                self.winner.set_result(winner)

    def getSeed():
        return [
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
            random.random() * 0x10000,
        ]

    #applies the (seed, action, action) list to the current game
    #clears out the queues so the game is ready to go
    #leave seed blank if you don't want to set the seed
    #(e.g. when following a real game)
    async def applyHistory(self, history):
        for seed, a1, a2 in history:
            if seed:
                await self.cmdQueue.put('>resetPRNG ' + str(seed))

            await self.p1Queue.get()
            await self.cmdQueue.put('>p1' + a1)

            await self.p2Queue.get()
            await self.cmdQueue.put('>p2' + a2)

