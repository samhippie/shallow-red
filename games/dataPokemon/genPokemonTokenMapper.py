#!/usr/bin/env python3

import json

#load showdown data
with open('pokedex.json', 'r') as f:
    data = json.load(f)

tokenMapper = {}

"""
#replace each instance of a pokemon with a basic summary
for id in data:
    mon = data[id]
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
    tokenMapper[species] = [species] + types + stats

"""
#split up level so we can see the number
for i in range(100):
    tokenMapper['l' + str(i)] = ['l', str(i)]
    
print(json.dumps(tokenMapper))
