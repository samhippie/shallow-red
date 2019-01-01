import asyncio
import copy
import random
import sys

#this is a simple one card poker variant AKA war
#http://www.cs.cmu.edu/~ggordon/poker/

numActions = 4

#don't need a context, so this is empty
class _Context:
    async def __aenter__(self):
        pass
    async def __aexit__(self, *args):
        pass

def getContext():
    return _Context()

def prettyPrintMove(move, req):
    return move

#our state machine for the game
class _Game():
    START = 0
    #dealing and ante are automatically done
    #p1 needs to bet
    P1_DEAL = 1
    #p1 inital bet of 0
    #p2 can either call or raise
    P2_CHECK = 2
    #p2 raised, p1 can match
    P1_RAISE = 3

    #p1 inital bet of 1
    #p2 can fold or call
    P2_CALL = 4

    #then the game ends
    END = 5

    #all possible actions
    DEAL = 'deal'
    FOLD = 'fold'
    CALL = 'call'
    RAISE = 'raise'
    
    actionDict = {
        P1_DEAL: [CALL, RAISE],#call is actually check but whatever
        P2_CHECK: [CALL, RAISE],
        P1_RAISE: [FOLD, CALL],
        P2_CALL: [FOLD, CALL],
        END: [],
    }

enumActionDict = {
    _Game.DEAL: 0,
    _Game.FOLD: 1,
    _Game.CALL: 2,
    _Game.RAISE: 3,
}
        
def enumAction(action):
    return enumActionDict[action]

def panic():
    print("ERROR THIS SHOULD NEVER HAPPEN", file=sys.stderr)
    quit()

def getSeed():
    return random.random()


class Game:
    def __init__(self, context=None, history=[[],[]], seed=None, verbose=False, file=sys.stdout):
        self.history = history
        self.seed = seed
        if seed:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()
        self.deck = list(range(13))
        self.file = file

        #this won't get set properly until the history is applied
        self.dealer = 0

        #already anted up
        self.pot = [1, 1]
        self.bet = 0
        self.hands = [0, 0]
        self.state = _Game.START

        loop = asyncio.get_event_loop()
        self.winner = loop.create_future()
        self._winner = None

        self.verbose = verbose
        self.infosets = [['start'],['start']]

    async def startGame(self):
        #dealer is determined by seed
        self.dealer = self.random.randrange(2)



        self.random.shuffle(self.deck)
        self.hands = [self.deck.pop(), self.deck.pop()]
        for i in range(2):
            self.infosets[i] += ['hand', str(self.hands[i])]

        if self.verbose:
            print('hands', self.hands, file=self.file)

        h = [copy.copy(self.history[0]), copy.copy(self.history[1])]
        while len(h[0]) or len(h[1]):
            player, req, actions = await self.getTurn()
            seed, action = h[player][0]
            del h[player][0]
            #ignore the seed, as the cards are already set
            await self.takeAction(player, req, action)

    async def getTurn(self):
        if self.state == _Game.START:
            return (self.dealer, {}, [_Game.DEAL])

        if self.state == _Game.END:
            if self._winner == None:
                #only hand is a high card
                self._winner = 0 if self.hands[0] > self.hands[1] else 1

            loser = (self._winner + 1) % 2
            winnings = self.pot[loser]
            if self.verbose:
                print('winner:', self._winner, 'winnings:', '$' + str(winnings), file=self.file)
            self.winner.set_result((self._winner, winnings))
            return (self._winner, {'win': winnings}, [])

        if self.state in [_Game.P1_DEAL, _Game.P1_RAISE]:
            player = (self.dealer + 1) % 2
        else:
            player = self.dealer

        actions = _Game.actionDict[self.state]

        return (player, {}, actions)


    def getInfoset(self, player):
        return self.infosets[player]

    #TODO we really should get rid of req
    async def takeAction(self, player, req, action):
        if self.verbose:
            print()
            print('state', self.state)
            print('player', player+1, 'takes action', action, file=self.file)
            print('bet:', self.bet, 'pot', self.pot, file=self.file)
        #all actions are public
        for i in range(2):
            #infosets are always in first person
            p = 0 if i == player else 1
            self.infosets[i] += [str(p), action]


        #I could probably simplify this by reducing the number of actions to 2
        #but then we lose some error detection
        if self.state == _Game.START:
            if action == _Game.DEAL:
                self.dealer = player
                self.state = _Game.P1_DEAL
            else:
                panic()
        elif self.state == _Game.P1_DEAL:
            if action == _Game.CALL:
                self.state = _Game.P2_CHECK
            elif action == _Game.RAISE:
                self.bet += 1
                self.pot[player] += self.bet
                self.state = _Game.P2_CALL
            else:
                panic()
        elif self.state == _Game.P2_CHECK:
            if action == _Game.CALL:
                self.state = _Game.END
            elif action == _Game.RAISE:
                self.bet += 1
                self.pot[player] += self.bet
                self.state = _Game.P1_RAISE
            else:
                panic()
        elif self.state == _Game.P1_RAISE:
            if action == _Game.FOLD:
                self.state = _Game.END
                self._winner = (player + 1) % 2
            elif action == _Game.CALL:
                self.pot[player] += self.bet
                self.state = _Game.END
            else:
                panic()
        elif self.state == _Game.P2_CALL:
            if action == _Game.FOLD:
                self.state = _Game.END
                self._winner = (player + 1) % 2
            elif action == _Game.CALL:
                self.pot[player] += self.bet
                self.state = _Game.END
            else:
                panic()
        else:
            panic()

