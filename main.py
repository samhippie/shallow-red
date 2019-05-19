#!/usr/bin/env python3

import argparse
import asyncio

import runner

async def main(args):
    await runner.trainAndPlay(args.numProcesses, args.pid, args.saveDir, args.clear)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a cfr process')
    parser.add_argument('numProcesses', type=int, help='Total number of processes')
    parser.add_argument('pid', type=int, help='Process id')
    parser.add_argument('-f', dest='saveDir', default=None, help='File to save models (will create if it does not exist)')
    parser.add_argument('-c', dest='clear', action='store_true', help='whether to clear saved samples')

    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
