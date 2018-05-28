#!/bin/bash

### als no-sync aanstaat gaat headless ook traag !!!! ####

trials = 1000
./bin/HFO --trials $trials  --no-logging --headless --fullstate --seed 123 --port=6000 --offense-agents=1 --defense-agents=1 --ball-x-min=0.25 --ball-x-max=0.25 --ball-y-min=0.0 --ball-y-max=0.0 --offense-on-ball 11 &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 10
python ./example/test_takers/basic_kicker.py --trials=$trials &
sleep 5
python ./example/test_keepers/deep_q_agent
sleep 10

#python ./example/test_keepers/evaluation.py --file1_size=5 --file2_size=5
#sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait


