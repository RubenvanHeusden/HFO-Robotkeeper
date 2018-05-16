#!/usr/bin/env python
# encoding: utf-8

# Before running this program, first Start HFO server:
# $> ./bin/HFO --offense-agents 1
from __future__ import division
import itertools
import math
import hfo
import argparse
from array import array
import numpy as np
import helper
from actionset import ActionSet



# this version of table based Q-Learning uses the high-level action
# set with the actions (CATCH, DEFEND_GOAL, GO_TO_BALL) and 9 possible
# bins in which the ball can be situated. 

# this very simple version is a first a attempt in developing some of the 
# core features that will be re-used in the neural network based version 
# of this algorithm




def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', default=6000, type=int)
  parser.add_argument('--feature_set', default=hfo.HIGH_LEVEL_FEATURE_SET, type=int)
  parser.add_argument('--trials', default=10000, type=int)
  args = parser.parse_args()
  state_space = 9 # amount of bins for ball to go in
  action_set = ActionSet("low_level")
  stats = {hfo.GOAL:0, hfo.CAPTURED_BY_DEFENSE:0, hfo.OUT_OF_BOUNDS:0, hfo.OUT_OF_TIME:0, hfo.SERVER_DOWN:0}

  

  alpha = 0.8
  gamma = 0.95
  stats_path = '/home/student/Desktop/HFO-master_ruben/example/test_keepers/stats.bin'
  advanced_stats = array('b')
  # Create the HFO Environment
  env = hfo.HFOEnvironment()  
  action_space = len(action_set)
  Q = np.zeros((state_space, action_space))
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
  
  
  
  env.connectToServer(args.feature_set,
                      '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', args.port,
                      'localhost', 'base_right', True)
                      
  
  x_bounds = [-1, 1]
  y_bounds = [-0.3, 0.3] 
  

  for episode in range(args.trials):
    s = helper.bin_ball_position(env.getState()[3:5], x_bounds, y_bounds)
    status = hfo.IN_GAME    
    while status == hfo.IN_GAME:
        a = np.argmax(Q[s, :] + np.random.randn(1, action_space)*(1./(episode+1)))
        env.act(*action_set[a])  
            
        status = env.step()
        features = env.getState()  
        reward = (1-np.linalg.norm(features[3:5]-features[0:2]))*10
        if status == hfo.GOAL :
            reward = -500
            break
        elif status == hfo.CAPTURED_BY_DEFENSE:
            reward = 500
            break
        elif status == hfo.OUT_OF_BOUNDS:
             reward = 200
             break
               
        s1 = helper.bin_ball_position(features[3:5], x_bounds, y_bounds)
        Q[s,a] = Q[s,a] + alpha*(reward + gamma*np.max(Q[s1,:]) - Q[s,a])
        
        s = s1
      
      # Grab the state features from the environment

      
      
      
      

      
      
      
      
      
      
    #advanced_stats.append(stats[status])
    print(('Episode %d ended with %s'%(episode, env.statusToString(status))))
    # Quit if the server goes down    
    if status == hfo.SERVER_DOWN:
      env.act(hfo.QUIT)
      exit()
      
  #print len(advanced_stats)
  #fp = open(stats_path, 'wb')
  #advanced_stats.tofile(fp)  
  #fp.close()    
  

if __name__ == '__main__':
  main()
  
  
  
