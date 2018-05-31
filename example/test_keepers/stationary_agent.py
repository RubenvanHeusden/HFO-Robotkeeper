#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1

import itertools
import math
import hfo
import argparse
from array import array
import numpy as np
import helper


# this version of table based Q-Learning uses the high-level action
# set with the actions (CATCH, DEFEND_GOAL, GO_TO_BALL) and 9 possible
# bins in which the ball can be situated. 

# this very simple version is a first a attempt in developing some of the 
# core features that will be re-used in the neural network based version 
# of this algorithm


    
def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', default=6000, type=int)
  parser.add_argument('--feature_set', default=hfo.LOW_LEVEL_FEATURE_SET, type=int)
  parser.add_argument('--trials', default=10000, type=int)
  args = parser.parse_args()

  env = hfo.HFOEnvironment()
  env.connectToServer(args.feature_set,
                      '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', args.port,
                      'localhost', 'base_right', True)
                      
  
  
  #num_features = hfo.getStateSize()
  
  for episode in range(args.trials):
    status = hfo.IN_GAME    
    while status != hfo.SERVER_DOWN:
        features = env.getState() 
        
        print math.degrees(math.asin(features[51]))               
        status = env.step()
        
        
              
    #advanced_stats.append(stats[status])
    print(('Episode %d ended with %s'%(episode, env.statusToString(status))))
    # Quit if the server goes down    
    if status == hfo.SERVER_DOWN:
      env.act(hfo.QUIT)
      exit()

if __name__ == '__main__':
  main()
  
  
  
