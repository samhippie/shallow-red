#!/usr/bin/env bash
#if you use this, you'll want to kill the main process first
for ((i=1;i<$1;i++))
do
    ./main.py $1 $i &
done
wait
