#!/bin/bash

### als no-sync aanstaat gaat headless ook traag !!!! ####

./bin/HFO  --no-sync --trials 10000 --fullstate  --seed 123 --port=6000 --offense-npcs=1 --defense-agents=1 --ball-x-min=0.0 --ball-x-max=0.0 --ball-y-min=0.0 --ball-y-max=0.0 --offense-on-ball 11 &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)
sleep 5
python ./example/test_keepers/silly_agent.py
sleep 10
#python ./example/test_keepers/evaluation.py --file1_size=5 --file2_size=5
#sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait


