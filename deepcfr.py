#!/usr/bin/env python3

import asyncio
import collections
import copy
import math
import numpy as np
import random
import sys
import torch
import torch.distributed as dist
import os.path

import config
import model
import dataStorage

#Deep MCCFR

#based on this paper
#https://arxiv.org/pdf/1811.00164.pdf

#this agent breaks some compatibility with the other agents
#so it needs a separate runner function
#(not that the other agents even exist any more)

class DeepCfrAgent:
    #each player gets one of each model
    #advModels calculates advantages
    #stratModels calculates average strategy

    #branch limit is the maximum number of actions taken
    #actions that aren't taken are probed (i.e. rolled out)

    #depth limit is the maximum number of turns taken from the root
    #after the limit it hit, all games are evaluated via rollout
    #this agent is only applied at the root, so depth limit might
    #significantly affect the quality of late-game strategies
    #(although we could use RM to find new late-game strategies,
    #but that's outside the scope of this agent)

    #resumeIter is the iteration to resume from
    #this really should be a parameter to the search function,
    #but we need to know about it when we initialize the models
    #this is a problem with our agent being made for a one-shot training cycle
    #instead of multiple training cycles like the others
    def __init__(
            self,
            writeLock,
            sharedDict,
            advModels=None, stratModels=None,
            singleDeep=False,
            verbose=False):

        self.pid = -1


        self.writeLock = writeLock
        #self.trainingBarrier = trainingBarrier
        self.sharedDict = sharedDict

        #if the adv models are passed in, assume we aren't responsible for sharing them
        if advModels:
            self.advModels = advModels
            self.manageSharedModels = False
        else:
            self.advModels = [full.model.DeepCfrModel(name='adv' + str(i), softmax=False, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]
            self.manageSharedModels = True

        #if stratModels:
            #self.stratModels = stratModels
        #else:
            #self.stratModels = [full.model.DeepCfrModel(name='strat' + str(i), softmax=True, writeLock=writeLock, sharedDict=sharedDict) for i in range(2)]

        #whether to save old models for single deep cfr
        self.singleDeep = singleDeep
        if(singleDeep):
            self.oldModels = [[],[]]
            self.oldModelWeights = [[],[]]

        #flag so if we never search, we don't bother training
        self.needsTraining = False

        self.verbose = verbose

    async def search(self, context, distGroup, pid=0, limit=100, innerLoops=1, seed=None, history=[[],[]]):
        self.pid = pid

        start = config.resumeIter if config.resumeIter else 0

        if self.pid == 0:
            print(end='', file=sys.stderr)
        for i in range(start, limit):

            #this is mainly used for setting a condition breakpoint
            #there's probably a better way
            #if i == 3:
                #print('ready for debugging')

            #for small games, this is necessary to get a decent number of samples
            for j in range(innerLoops):
                if self.pid == 0:
                    print('\rTurn Progress: ' + str(i) + '/' + str(limit) + ' inner ' + str(j) + '/' + str(innerLoops), end='', file=sys.stderr)
                self.needsTraining = True
                #we want each game tree traversal to use the same seed
                if seed:
                    curSeed = seed
                else:
                    curSeed = config.game.getSeed()
                game = config.game.Game(context=context, seed=curSeed, history=history, verbose=self.verbose)
                await game.startGame()
                await self.cfrRecur(context, game, curSeed, history, i)


            #save our adv data after each iteration
            #so the non-zero pid workers don't have data cached
            self.advModels[i % 2].clearSampleCache()
            #go ahead and clear our strat caches as well
            #just in case the program is exited
            #for j in range(2):
                #self.stratModels[j].clearSampleCache()

            #only need to train about once per iteration
            #self.trainingBarrier.wait()

            #this doesn't actually train
            #if self.pid != 0:
                #print('attempting to send train')
                #self.advTrain(i % 2, iter=i // 2 + 1)
            #print(self.pid, 'about to hit training barrier')
            dist.barrier(distGroup)
            #print(self.pid, 'just hit training barrier')
            if self.pid == 0:
                if self.needsTraining:
                    print('sending train message')
                    self.advTrain(i % 2, iter=i // 2 + 1)
                #if self.manageSharedModels:
                #self.advModels[i % 2].net.cpu()
                #self.sharedDict['advNet' + str(i % 2)] = self.advModels[i % 2].net
                #self.advModels[i % 2].net.cuda()
                #else:
                    #self.advModels[i % 2].shareMemory()

            distGroup.barrier()
            #0 to continue, 1 to stop early
            if os.path.isfile('stopEarly'):
                #cant' rename the file here
                if self.pid == 0:
                    print('stopping early')
                break

            self.needsTraining = False

            if self.pid == 0:
                print('\nplaying games', file=sys.stderr)
            
        distGroup.barrier()

        if self.pid == 0:
            #have to wait to rename this until all processes have had a chance to see it
            if os.path.isfile('stopEarly'):
                os.rename('stopEarly', 'XstopEarly')
            out = torch.zeros(3)
            dist.send(out, dst=0)
            print('playtime is over', file=sys.stderr)
            print(file=sys.stderr)

    def advTrain(self, player, iter=1):
        #send message to net process to train network
        dist.send(torch.tensor([1, player, 0]), dst=0)
        #just block until it's done
        out = torch.zeros(1, dtype=torch.long)
        dist.recv(out, src=0)

        #model = self.advModels[player]
        #model.train(epochs=config.advEpochs)
        #if(self.singleDeep):
            #model.net.cpu()
            #self.oldModels[player].append(model.net)
            #model.net.cuda()
            #self.oldModelWeights[player].append(iter)
            #self.sharedDict['oldModels'] = self.oldModels
            #self.sharedDict['oldModelWeights'] = self.oldModelWeights

        
    def stratTrain(self):
        if(self.singleDeep):
            self.oldModels = self.sharedDict['oldModels']
            self.oldModelWeights = self.sharedDict['oldModelWeights']
            print('skipping strategy', file=sys.stderr)
            return
        if self.pid == 0:
            print('training strategy', file=sys.stderr)
        #we train both strat models at once
        for i, model in enumerate(self.stratModels):
            #let's try copying the embedding from the current adv net
            model.net.embeddings = self.advModels[i].net.embeddings
            model.train(epochs=config.stratEpochs)

    def getPredict(self, player, infoset):
        inputTensor = model.infosetToTensor(infoset)
        #if self.pid == 1:
            #print('sending', inputTensor)
        #print(self.pid, 'sending request')
        #print(self.pid, 'sending', inputTensor)
        dist.send(torch.tensor([2, player, inputTensor.shape[0]]), dst=0)
        #print(self.pid, 'sending input with shape', inputTensor.shape, 'dtype', inputTensor.dtype, inputTensor)
        dist.send(inputTensor, dst=0)
        out = torch.zeros(config.game.numActions + 1)
        #print(self.pid, 'getting output')
        dist.recv(out, src=0)
        #print(self.pid, 'got output')
        out = out.detach().numpy()
        #if self.pid == 1:
            #print('got', out)
        return out[0:-1], out[-1]

    #getting probability for a given model to follow a given trajectory
    #where a trajectory is a list of infoset-action pairs
    def getReachProb(self, model, traj):
        reachProb = 1
        for infoset, action in traj:
            probs, _ = model.predict(infoset)
            actionNum = config.game.enumAction(action)
            for i, p in enumerate(probs):
                if probs[i] < 0:
                    probs[i] = 0
            pSum = sum(probs)
            if pSum == 0:
                #this should be 0, but I don't want to divide by zero somewhere down the line
                reachProb *= 0.001
            else:
                reachProb *= probs[actionNum] / pSum
        return reachProb

    #getting final probabilities for executing a strategy
    def getProbs(self, player, infoset, actions, prevTrajectory=None):
        if(self.singleDeep):
            stratProbs = None
            expVal = 0
            weights = []
            model = self.advModels[player]
            for i in range(len(self.oldModels[player])):
                model.net = self.oldModels[player][i]
                weight = self.oldModelWeights[player][i]
                reachProb = self.getReachProb(model, prevTrajectory)
                weight *= reachProb
                probs, ev = model.predict(infoset, trace=False)
                for i, p in enumerate(probs):
                    if p < 0:
                        probs[i] = 0
                pSum = sum(probs)
                if pSum > 0:
                    probs /= pSum
                else:
                    probs = np.array([1 / len(probs) for p in probs])
                #probs, ev = self.getPredict(player, infoset)
                expVal += ev# * weight
                if(stratProbs is not None):
                    stratProbs += weight * probs
                else:
                    stratProbs = weight * probs

            expVal /= len(self.oldModels[player])
        else:
            sm = self.stratModels[player]
            stratProbs, expVal = sm.predict(infoset, trace=False)
            #stratProbs, expVal = self.getPredict(sm, infoset)
        print('infoset', infoset)
        print('strat probs', stratProbs)
        print('expVal', expVal)
        actionNums = [config.game.enumAction(a) for a in actions]
        probs = []
        for n in actionNums:
            probs.append(stratProbs[n])
        probs = np.array(probs)

        pSum = np.sum(probs)
        if pSum > 0:
            return probs / np.sum(probs)
        else:
            #play randomly
            return np.array([1 / len(actions) for a in actions])

    #recursive implementation of cfr
    #history is a list of (seed, player, action) tuples
    #assumes the game has already had the history applied
    async def cfrRecur(self, context, game, startSeed, history, iter, depth=0, q=1, rollout=False):
        if config.depthLimit and depth > config.depthLimit:
            rollout = True

        onPlayer = iter % 2
        offPlayer = (iter + 1) % 2

        player, req, actions = await game.getTurn()

        if 'win' in req:
            if player == onPlayer:
                #return (req['win'] + 2) / 4
                return req['win']
            else:
                return -1 * req['win']

        #game uses append, so we have to make a copy to keep everything consistent when we get advantages later
        infoset = copy.copy(game.getInfoset(player))

        if player == offPlayer:
            #get probs so we can sample a single action
            probs, _ = self.regretMatch(offPlayer, infoset, actions, -1)
            exploreProbs = probs * (1 - config.offExploreRate) + config.offExploreRate / len(actions)
            action = np.random.choice(actions, p=exploreProbs)

            #if depth == 1 and self.pid == 0:
                #print('offplayer ' + str(player) + ' hand ' + str(game.hands[player]) + ' probs', list(zip(actions, probs)), file=sys.stderr)
            await game.takeAction(player, action)

            if player == 0:
                newHistory = [history[0] + [(None, action)], history[1]]
            else:
                newHistory = [history[0], history[1] + [(None, action)]]

            onExpValue = await self.cfrRecur(context, game, startSeed, newHistory, iter, depth=depth, rollout=rollout, q=q)

            #save sample for final average strategy
            """
            if not rollout:
                sm = self.stratModels[offPlayer]
                sm.addSample(infoset, zip(actions, probs), iter // 2 + 1, -1 * onExpValue)
            """
            return onExpValue

        elif player == onPlayer:
            #get probs, which action we take depends on the configuration
            probs, regrets = self.regretMatch(onPlayer, infoset, actions, depth)
            #if depth == 1 and self.pid == 0:
                #print('onplayer ' + str(player) + ' hand ' + str(game.hands[player]) + ' probs', list(zip(actions, probs)), 'advs', regrets, file=sys.stderr)
            if rollout:
                #we pick one action according to the current strategy
                #like this paper, except we also do it when we hit a depth limit
                #https://poker.cs.ualberta.ca/publications/AAAI12-generalmccfr.pdf
                actions = [np.random.choice(actions, p=probs)]
                actionIndices = [0]
            elif config.branchingLimit:
                #select a set of actions to pick
                #chance to play randomly instead of picking the best actions
                #this paper suggests playing according the currect strategoy with some exploration factor for outcome
                #sampling (i.e. branchLimit = 1), so I assume that
                #http://mlanctot.info/files/papers/nips09mccfr.pdf
                exploreProbs = probs * (1 - config.onExploreRate) + config.onExploreRate / len(probs)
                actionIndices = np.random.choice(len(actions), min(len(actions), config.branchingLimit), 
                        replace=False, p=exploreProbs)
            else:
                #we're picking every action
                actionIndices = list(range(len(actions)))

            #get expected reward for each action
            rewards = []
            gameUsed = False

            for i in range(len(actions)):
                action = actions[i]

                #use rollout for non-sampled actions
                if not i in actionIndices:
                    curRollout = True
                else:
                    curRollout = rollout

                #don't have to re-init game for the first action
                if gameUsed:
                    game = config.game.Game(context, seed=startSeed, history=history, verbose=self.verbose)
                    await game.startGame()
                    await game.getTurn()
                else:
                    gameUsed = True

                #I want to see if we get good results by keeping the RNG the same
                #this is closer to normal external sampling
                #seed = await game.resetSeed()
                await game.takeAction(player, action)
                #historyEntry = (None, player, action)

                if player == 0:
                    newHistory = [history[0] + [(None, action)], history[1]]
                else:
                    newHistory = [history[0], history[1] + [(None, action)]]

                r = await self.cfrRecur(context, game, startSeed, newHistory, iter, depth=depth+1, rollout=curRollout, q=q*probs[i])
                rewards.append(r)

            if not rollout:
                #save sample of advantages
                stateExpValue = 0
                for p,r in zip(probs, rewards):
                    stateExpValue += p * r
                advantages = [r - stateExpValue for r in rewards]
                #if self.pid == 0:
                    #print('infoset', infoset)
                    #print('actions, prob, reward, advantage', *list(zip(actions, probs, rewards, advantages)))
                #CFR+, anyone?
                #also using the sqrt(t) equation from that double neural cfr paper
                #advantages = [max(0, math.sqrt(iter // 2) * g / math.sqrt(iter // 2 + 1) + (r - stateExpValue) / math.sqrt(iter // 2 + 1)) for r, g in zip(rewards, regrets)]
                #if depth == 1 and self.pid == 0:
                    #print('onplayer', player, 'hand', game.hands[player], 'new advs', list(zip(actions, advantages)), 'exp value', stateExpValue, file=sys.stderr)
                #print('advantages', advantages)

                am = self.advModels[onPlayer]
                am.addSample(infoset, zip(actions, advantages), iter // 2 + 1, stateExpValue)

                #if depth == 0 and self.pid == 0:
                    #print('player', str(onPlayer), file=sys.stderr)
                    #print('stateExpValue', stateExpValue, 'from', list(zip(probs, rewards)), file=sys.stderr)
                    #print('advantages', list(zip(actions, advantages)), file=sys.stderr)

                return stateExpValue
            else:
                #we can't calculate advantage, so we can't update anything
                #we only have one reward, so just return it
                return rewards[0]

   
    #generates probabilities for each action
    #based on modeled advantages
    def regretMatch(self, player, infoset, actions, depth):
        #am = self.advModels[player]
        #advs, expVal = am.predict(infoset)
        advs, expVal = self.getPredict(player, infoset)
        #illegal actions should be 0
        flatAdvs = np.zeros(len(advs))
        actionNums = [config.game.enumAction(a) for a in actions]
        probs = []
        for n in actionNums:
            probs.append(max(0, advs[n]))
            flatAdvs[n] = advs[n]
        #if depth == 0 and self.pid == 0:
            #print('predicted advantages', [(action, advs[n]) for action, n in zip(actions, actionNums)], file=sys.stderr)
        probs = np.array(probs)
        pSum = np.sum(probs)
        if pSum > 0:
            return probs / pSum, flatAdvs
        else:
            #pick the best action with probability 1
            best = None
            for i in range(len(actionNums)):
                n = actionNums[i]
                if best == None or advs[n] > advs[actionNums[best]]:
                    best = i
            probs = [0 for a in actions]
            probs[best] = 1
            return np.array(probs), flatAdvs
            #actually, play randomly
            #return np.array([1 / len(actions) for a in actions]), flatAdvs
