import asyncio
import copy
import random
import sys

numActions = 4

#don't need context
class _Context:
    async def __aenter__(self):
        pass
    async def __aexit__(self, *args):
        pass

def getContext():
    return _Context()

def prettyPrintMove(move, req=None):
    return move

class _Game:

    DEAL = 'deal'
    FOLD = 'fold'
    CALL = 'call'
    RAISE = 'raise'

#TODO
