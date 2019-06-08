#!/usr/bin/env python3

import json
import re

#load showdown data
with open('pokedex.json', 'r') as f:
    dex = json.load(f)
with open('moves.json', 'r') as f:
    moves = json.load(f)

tokenMapper = {}

#for both moves and pokemon, some things use the id instead of species/name
#e.g. icebeam instead of ice beam
#so I just map both to the same thing

#replace each instance of a pokemon with a basic summary
for id in dex:
    mon = dex[id]
    species = mon['species'].lower()
    types = [t.lower() for t in mon['types']]
    #single type -> two of same type
    if len(types) == 1:
        types.append(types[0])
    base = mon['baseStats']
    stats = [
        str(base['hp']),
        str(base['atk']),
        str(base['def']),
        str(base['spa']),
        str(base['spd']),
        str(base['spe']),
    ]
    tokenMapper[species] = [id] + types# + stats
    tokenMapper[id] = [id] + types# + stats

#replace each instance of a move with basic summary
for id in moves:
    move = moves[id]
    name = move['name'].lower()
    stripName = re.sub(r'[^a-z0-9]', '', name)
    type = move['type'].lower()
    cat = move['category'].lower()
    power = str(move['basePower'])
    tokenMapper[name] = [id, type, cat, power]
    tokenMapper[id] = [id, type, cat, power]


#split up level so we can see the number
for i in range(100):
    tokenMapper['l' + str(i)] = ['l', str(i)]
    
print(json.dumps(tokenMapper))
