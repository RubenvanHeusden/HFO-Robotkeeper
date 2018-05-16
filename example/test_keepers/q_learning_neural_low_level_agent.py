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
import tensorflow as tf
import random
from lowlevelactionset import LowLevelActionSet


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
  
  # Starting the TensorFlow network
  tf.reset_default_graph()

 
  # Create the HFO Environment
  env = hfo.HFOEnvironment()
  feature_space_n = 100
  action_space_n = 8
  # Connect to the server with the specified
  # feature set. See feature sets in hfo.py/hfo.hpp.
   
  env.connectToServer(args.feature_set,
                      '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', args.port,
                      'localhost', 'base_right', True)
     
  # setting feed-forward part of the network
  
  inputs1 = tf.placeholder(shape=[1,feature_space_n],dtype=tf.float32)
  W = tf.Variable(tf.random_uniform([feature_space_n,action_space_n],0,0.01))
  Qout = tf.matmul(inputs1,W)
  predict = tf.argmax(Qout,1)
  
  nextQ = tf.placeholder(shape=[1,action_space_n],dtype=tf.float32)
  loss = tf.reduce_sum(tf.square(nextQ - Qout))
  trainer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
  updateModel = trainer.minimize(loss)
  
  init = tf.initialize_all_variables()
  action_set = LowLevelActionSet()
  
  
  
  gamma = 0.99
  e = 0.1
  
  x_bounds = [-1, 1]
  y_bounds = [-0.3, 0.3]   
    
  #num_features = hfo.getStateSize()
  with tf.Session() as sess:
    sess.run(init)
  
    for episode in range(args.trials):
        s = helper.bin_ball_position(env.getState()[3:5], x_bounds, y_bounds)
        status = hfo.IN_GAME    
        while status == hfo.IN_GAME:
            a,allQ = sess.run([predict,Qout],feed_dict={inputs1:np.identity(feature_space_n)[s:s+1]})
            if np.random.rand(1) < e:
                a[0] = random.randint(0, 8)
            #Get new state and reward from environment
            env.act(*action_set[a[0]])
            status = env.step()  
            s1 = helper.bin_ball_position(env.getState()[3:5], x_bounds, y_bounds)
            reward = (1-np.linalg.norm(env.getState()[3:5]-env.getState()[0:2]))*100
            if status == hfo.GOAL :
                reward = -500
                e = 1./((episode/50) + 10)
                break
            elif status == hfo.CAPTURED_BY_DEFENSE:
                e = 1./((episode/50) + 10)
                reward = 500
                break
            #Obtain the Q' values by feeding the new state through our network
            Q1 = sess.run(Qout,feed_dict={inputs1:np.identity(feature_space_n)[s1:s1+1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,a[0]] = reward + gamma*maxQ1
            #Train our network using target and predicted Q values
            _,W1 = sess.run([updateModel,W],feed_dict={inputs1:np.identity(feature_space_n)[s:s+1],nextQ:targetQ})
            s = s1
        
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
  
  
  
