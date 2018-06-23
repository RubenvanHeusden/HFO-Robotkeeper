#!/bin/bash

### als no-sync aanstaat gaat headless ook traag !!!! ####

trials=20000
./bin/HFO --trials $trials --headless --no-logging --fullstate --seed 123 --port=6000 --offense-agents=1 --defense-agents=1 --ball-x-min=0.80 --ball-x-max=0.80 --ball-y-min=0.0 --ball-y-max=0.0 --offense-on-ball 11 &
# Sleep is needed to make sure doesn't get connected too soon, as unum 1 (goalie)

sleep 10
python ./example/test_takers/basic_kicker.py --trials=$trials &
sleep 5
python ./example/test_keepers/fixed_keeper/fixed_agent.py --filename="last/phase1" --trials=$trials

#python ./example/test_keepers/evaluation.py --file1_size=5 --file2_size=5
#sleep 5
# The magic line
#   $$ holds the PID for this script
#   Negation means kill by process group id instead of PID
trap "kill -TERM -$$" SIGINT
wait


