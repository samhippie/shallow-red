#!/usr/bin/env python3

#hard coded moveset
#not all of these are legal, but we can deal with that later
moveSet = []
#no switching in 1v1
for i in range(4):
    for extra in ['', ' mega', ' zmove']:
        moveSet.append(' move ' + str(i+1) + extra)

#hard coded team preview combinations
#for 1v1, we only need to specify 1 mon out of 3
teamSet = []
for i in range(3):
    teamSet.append(' team ' + str(i+1))

