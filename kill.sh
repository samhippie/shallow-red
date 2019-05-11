#!/bin/sh
#Ctrl+c leaves some processes, including some that use ports
kill $(ps aux | grep -e 'main\.py' | grep -v grep | awk '{print $2}')
