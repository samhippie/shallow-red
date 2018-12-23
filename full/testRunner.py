#!/usr/bin/env python3

import asyncio
import random

import full.game
from deep.deepRunner import getPSProcess

#this is for playing random games
#no AI, but it's good for testing

async def randomGame(ps, history=[[],[]]):
    game = full.game.Game(ps, format='1v1', history=history, verbose=True)
    await game.startGame()

    async def play():
        while True:
            player, req, actions = await game.getTurn()
            action = random.choice(actions)
            await game.takeAction(player, req, action)

    playTask = asyncio.ensure_future(play())
    winner = await game.winner
    playTask.cancel()
    print('winnner:', winner)

