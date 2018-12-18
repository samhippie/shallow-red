#!/usr/bin/env python3

import asyncio
import random

import deep.game
from deep.deepRunner import getPSProcess

#this is for playing random games
#no AI, but it's good for testing

async def randomGame(ps):
    game = deep.game.Game(ps, format='1v1', verbose=True)
    await game.startGame()

    async def play():
        while True:
            player, req, actions = await game.getTurn()
            action = random.choice(actions)
            await game.takeAction(player, req, action)

    playTask = asyncio.ensure_future(play())
    winner = await game.winner
    playTask.cancel()
    print('winner:', winner)

