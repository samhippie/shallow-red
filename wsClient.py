#!/usr/bin/env python3

#this connects to a known PS server
#logs in
#sets its team
#accepts challenges
#reads requests
#uses user.py's one-turn search to pick moves

import asyncio
import json
import requests
import urllib
import websockets

from user import InteractivePlayer

teams = [
    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|F||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,|F|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,icebeam|Quiet|252,4,,252,,|F||||',

    '|charmander|lifeorb||flareblitz,brickbreak,dragondance,outrage|Adamant|,252,,,4,252|M||||]|bulbasaur|chestoberry||gigadrain,toxic,sludgebomb,rest|Quiet|252,4,,252,,|M|,0,,,,|||]|squirtle|leftovers||fakeout,aquajet,hydropump,icebeam|Quiet|252,4,,252,,|M||||',
]

def getLogin():
    with open('secret.txt', 'r') as file:
        line = file.readline()
        user = line.split(':')[0]
        passwd = line.split(':')[1][0:-1]#no newlines
    return user, passwd

#this handles input from the server
#so we can just await on whatever piece of data we want
class SocketReceiver:
    def __init__(self, ws):
        self.ws = ws
        self.loop = asyncio.get_event_loop()
        self.challstr = self.loop.create_future()
        self.challenge = self.loop.create_future()
        self.request = self.loop.create_future()
        self.win = self.loop.create_future()

    #call this in between games
    def gameReset(self):
        self.challenge = self.loop.create_future()
        self.request = self.loop.create_future()
        self.win = self.loop.create_future()

    async def startLoop(self):
        asyncio.ensure_future(self.recvLoop())

    async def recvLoop(self):
        running = True
        while running:
            msg = await self.ws.recv()
            print('got msg', msg)
            data = msg.split('|')
            if data[1] == 'challstr':
                self.challstr.set_result(msg[10:])
            elif data[1] == 'updatechallenges':
                print('got challenge msg')
                #got a challenge
                challenges = json.loads(data[2])
                names = list(challenges['challengesFrom'])
                print(names)
                #only save the first challenge
                if len(names) > 0:
                    print(self.challenge)
                    #assume nothing is waiting on it
                    if self.challenge.done():
                        self.challenge = self.loop.create_future()
                    self.challenge.set_result(names[0])
            elif data[1] == 'request':
                #ignore empty requests
                if data[2]:
                    room = data[0]
                    #sometimes (always?) rooms start with '>' in messages
                    if room[0] == '>':
                        room = room[1:]
                    #sometimes (always?) rooms end with \n
                    if room[-1] == '\n':
                        room = room[0:-1]
                    request = json.loads(data[2])
                    self.request.set_result((room, request))
            elif len(data[1].strip()) == 0 and data[-2] == 'win':
                room = data[0]
                #sometimes (always?) rooms start with '>' in messages
                if room[0] == '>':
                    room = room[1:]
                #sometimes (always?) rooms end with \n
                if room[-1] == '\n':
                    room = room[0:-1]
                result = data[-1]
                if result[-1] == '\n':
                    result = result[:-1]
                print('trying to leave', room)
                self.win.set_result((room, result))

    #waits for the next challenge
    #isn't designed to work well with multiple challenges
    #returns the name i.e. what you use with /accept
    async def getChallenge(self):
        challenge = await self.challenge
        print('got result for challenge future')
        self.challenge = self.loop.create_future()
        return challenge

    async def getRequest(self):
        request = await self.request
        self.request = self.loop.create_future()
        return request

    def hasWin(self):
        return self.win.done()

    async def getWin(self):
        room, result = await self.win
        self.win = self.loop.create_future()
        self.gameReset()
        return room, result

#assumes we're in one battle at a time
async def doBattle(ws, sr):
    player = InteractivePlayer(teams, limit=100, numProcesses=1, format='1v1')
    async def gameLoop():
        hasSentMessage = False
        while True:
            room, request = await sr.getRequest()
            if not hasSentMessage:
                await ws.send(room + '|bee boo boo bop boo boo beep')
                hasSentMessage = True
            action = await player.getAction(request)
            await ws.send(room + '|/' + action.strip())
    gameTask = asyncio.ensure_future(gameLoop())
    room, result = await sr.getWin()
    #hardcoded username, I don't really care
    if result == 'ShallowRed':
        await ws.send(room + '|learn to play')
    else:
        await ws.send(room + '|i\'d win in bo3')
    gameTask.cancel()
    print('leaving')
    await ws.send('|/leave ' + room)



async def main():
    user, passwd = getLogin()
    #connect to our modified server
    print('connecting')
    async with websockets.connect('ws://localhost:8000/showdown/websocket') as websocket:
        print('starting')
        sr = SocketReceiver(websocket)
        await sr.startLoop()
        challstr = await sr.challstr
        print('got chall')
        #make a request to log in
        payload = {'act': 'login', 'name': user, 'pass': passwd, 'challstr': challstr}
        result = requests.post('https://play.pokemonshowdown.com/action.php', data=payload)
        loginData = json.loads(result.text[1:])
        print('logged in')

        user = loginData['curuser']['username']
        assertion = loginData['assertion']
        await websocket.send('|/trn ' + user + ',0,' + assertion)
        print('joining lobby')
        await websocket.send('|/join lobby')
        print('setting team')
        #bot is player 2
        await websocket.send('|/utm ' + teams[1])
        while True:
            chalName = await sr.getChallenge()
            await websocket.send('|/accept ' + chalName)
            await doBattle(websocket, sr)
            print('done with battle')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
