#!/usr/bin/env python3

import asyncio
import random

import full.game
from deep.deepRunner import getPSProcess

#this is for playing random games
#no AI, but it's good for testing

async def randomGame(ps):
    game = full.game.Game(ps, format='1v1', verbose=False)
    await game.startGame()

    async def play():
        while True:
            player, req, actions = await game.getTurn()
            action = random.choice(actions)
            await game.takeAction(player, req, action)

    playTask = asyncio.ensure_future(play())
    winner = await game.winner
    playTask.cancel()
    #read current tokens
    with open('/home/sam/data/vocab.txt','r') as file:
        vocab = {line[:-1] for line in file.readlines()}
    #add own tokens
    for token in game.infosets[0] + game.infosets[1]:
        vocab.add(token)
    print('vocab size:', len(vocab))
    #write new tokens back out
    with open('/home/sam/data/vocab.txt', 'w') as file:
        for token in vocab:
            print(token, file=file)

