#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import itertools
import math
from hfo import *
import argparse


"""
Possible game statuses:
  [IN_GAME] Game is currently active
  [GOAL] A goal has been scored by the offense
  [CAPTURED_BY_DEFENSE] The defense has captured the ball
  [OUT_OF_BOUNDS] Ball has gone out of bounds
  [OUT_OF_TIME] Trial has ended due to time limit
  [SERVER_DOWN] Server is not alive
"""


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', default=6000, type=int)
  parser.add_argument('--feature_set', default=HIGH_LEVEL_FEATURE_SET, type=int)
  args = parser.parse_args()
  
  # Create the HFO Environment
  hfo = HFOEnvironment()
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  hfo.connectToServer(args.feature_set,
                      '/home/student/Desktop/HFO-master/bin/teams/base/config/formations-dt', args.port,
                      'localhost', 'base_right', True)
                      
  for episode in range(5):
    status = IN_GAME
    while status == IN_GAME:
      # Grab the state features from the environment
      features = hfo.getState()
      status = hfo.step()
    # Check the outcome of the episode
    print(('Episode %d ended with %s'%(episode, hfo.statusToString(status))))
    # Quit if the server goes down
    if status == SERVER_DOWN:
      hfo.act(QUIT)
      exit()

if __name__ == '__main__':
  main()
  
  
  
