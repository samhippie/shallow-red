#!/usr/bin/env python3

import asyncio
import random

import config

#this is for playing random games
#no AI, but it's good for testing

async def randomGameImpl(context):
    game = config.game.Game(context, history=config.GameConfig.history, verbose=True)
    await game.startGame()

    async def play():
        while True:
            player, req, actions = await game.getTurn()
            print('player', player+1, 'infoset', game.getInfoset(player))
            action = random.choice(actions)
            await game.takeAction(player, req, action)

    playTask = asyncio.ensure_future(play())
    winner = await game.winner
    playTask.cancel()
    print('winnner:', winner)

async def randomGame(context=None):
    if not context:
        async with config.game.getContext() as context:
            await randomGameImpl(context)
    else:
        await randomGameImpl(context)

