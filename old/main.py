#!/usr/bin/env python3

import asyncio
import collections
from contextlib import suppress
import copy
import math
import numpy as np
import os
import random
import sys
import subprocess

import full.runner

async def main():
    format = '1v1'
    #format = '2v2doubles'
    #format='singles'
    #format='vgc'

    await full.runner.playTestGame(format=format, limit=200, numProcesses=8, advEpochs=3000, stratEpochs=100000, branchingLimit=3, depthLimit=10, innerLoops=20, resumeIter=None)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
