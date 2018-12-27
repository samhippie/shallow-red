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

    await full.runner.playTestGame(format=format, limit=50, numProcesses=16, advEpochs=1000, stratEpochs=10000, branchingLimit=3, depthLimit=5, innerLoops=5, resumeIter=None)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
