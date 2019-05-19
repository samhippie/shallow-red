
import time
import torch.multiprocessing as mp
import torch.distributed as dist
import torch

import config
import model


#runs the main process that controls the network
#mainly sends and receives messages
#and tells the network to do stuff

#is 'semaphore' the right word? who cares
SEMAPHORE = 10
class Receiver:
    def __init__(self, rank, defaultShape, defaultDType):
        self.rank = rank
        self.tensor = None
        self.req = None
        self.defaultShape = defaultShape
        self.defaultDType = defaultDType

    def isReady(self, prefix=None):
        if prefix is None:
            return not (self.tensor is None or self.tensor[0].item() == SEMAPHORE)
        else:
            return self.tensor is not None and all([self.tensor[i].item() == v for i,v in enumerate(prefix)])

    def get(self):
        self.req.wait()
        return self.tensor

    def irecv(self, shape=None, dtype=None):
        if shape is None:
            shape = self.defaultShape
        if dtype is None:
            dtype = self.defaultDType
        self.tensor = torch.zeros(shape, dtype=dtype)
        self.tensor[0] = SEMAPHORE
        self.req = dist.irecv(self.tensor, src=self.rank)

class MassReceiver:
    def __init__(self, numProcesses, defaultShape, defaultDType):
        self.receivers = []
        self.removed = []
        self.defaultShape = defaultShape
        self.defaultDType = defaultDType
        for i in range(1, numProcesses):
            rec = Receiver(i, defaultShape, defaultDType)
            rec.irecv(defaultShape)
            self.receivers.append(rec)

        self.recvIndex = 0

    #gets next receiver that has a value
    def getNextPending(self, prefix=None):
        #check if any receivers are ready
        for i in range(len(self.receivers)):
            #start after last checked to avoid starvation
            idx = (self.recvIndex + 1 + i) % len(self.receivers)
            if self.receivers[idx].isReady(prefix):
                self.recvIndex = idx
                return self.receivers[idx]
        return None

    #gets next receiver, regardless of status
    #don't use this, it won't work if we have a pending irecv from an agent
    def getNext(self):
        self.recvIndex = (self.recvIndex + 1) % len(self.receivers)
        return self.receivers[self.recvIndex]

    #removes receiver from consideration
    def remove(self, rank):
        i, rec = next((i,rec) for i,rec in enumerate(self.receivers) if rec.rank == rank)
        del self.receivers[i]
        self.removed.append(rec)

    #puts all receivers back into consideration
    def restoreAll(self):
        for rec in self.removed:
            self.receivers.append(rec)
        self.removed = []
        



#send 0, X, X to stop
#send 1, player, X to train net
#send 2, player, size to request evaluation, where player is which player and size is the size of your network input
#   follow with network input
#   then recv the results
#10 is reserved
async def run(agent, numProcesses, playTestGames=None, numTestGames=0):
    receiver = MassReceiver(numProcesses, 3, torch.long)

    while True:
        rec = receiver.getNextPending()
        while not rec:
            rec = receiver.getNextPending()
            #rec = receiver.getNext()
        #else:
            #print('rec', rec.rank, rec.tensor)
        msg = rec.get()
        rank = rec.rank

        #print(rank, msg)

        #stop
        if msg[0].item() == 0:
            break

        #train a new network
        elif msg[0].item() == 1:
            rec.irecv()
            if rank != 1:
                #receiver.remove(rank)
                pass
            else:
                player = msg[1].item()
                #train
                agent.advModels[player].clearSampleCache()
                agent.advModels[player].train(iteration=len(agent.oldModels[player]), epochs=config.advEpochs)
                #save model for later evaluation
                agent.oldModels[player].append(agent.advModels[player].net)
                #assume we train once per iteration, so len should equal iteration number
                agent.oldModelWeights[player].append(len(agent.oldModels[player]))

                if len(agent.oldModels[0]) > 0 and len(agent.oldModels[1]) > 0 and playTestGames:
                    print('playing test games')
                    with open(config.progressGamePath + 'progress' + str(len(agent.oldModels[player])) + '-' + config.gameName + '-' + str(round(time.time())) + '.txt', 'w') as f:
                        await playTestGames(agent, numTestGames, f)
                else:
                    print('not playing test games')

                #let agent know we're done
                out = torch.tensor([1], dtype=torch.long)
                dist.isend(out, dst=rank)

        #evaluate the network
        elif msg[0].item() == 2:
            inputs = []
            ranks = []
            #batch any pending network requests
            nextRec = rec
            while nextRec:
                #print(nextRec.rank, 'next rec', msg)
                #get metadata
                player = msg[1].item()
                inputSize = msg[2].item()
                #get network input
                netInput = torch.zeros((inputSize, 1 + model.NUM_TOKEN_BITS), dtype=torch.long)
                dist.recv(netInput, src=nextRec.rank)
                #print(nextRec.rank, 'got net input', netInput)
                #save for evaluation
                inputs.append(netInput)
                ranks.append(nextRec.rank)

                #print('sending irecv')
                nextRec.irecv()
                #print('getting next next rec')
                nextRec = receiver.getNextPending([2, player])
                if nextRec:
                    #print('got next next rec')
                    msg = nextRec.get()


            #print('running through network')
            outs = agent.advModels[player].batchPredict(inputs, convertToTensor=False).cpu()
            for out, dst in zip(outs, ranks):
                #print('sending eval to', dst)
                dist.send(out, dst=dst)
