# This is an attempt to create a keeper agent that is stationary on the 
# goal line and just stops a single straight through shot by moving 
# in a discrete action space


import random
import hfo
import numpy as np
import tensorflow as tf
from collections import deque

def bin_states(num_bins, pos):
    pass


def get_reward(state, status):
    pass



class Memory:
    def __init__(self):
        pass


































# Neural network used for the fixed line keeper experiment
# Vanilla, DQN, DOUBLE Q, DUELLING Q
class NeuralNetwork:
    
    def __init__(self, session, num_inputs, num_outputs):
    
        self.sess = session
        self.layer1_size = 128
        self.layer2_size = 64
        self.layer3_size = 32
        self.layer4_size = 16

        self.learning_rate = 0.0001
        
        self._gauss_init = tf.truncated_normal_initializer(mean=0.0, 
                                            stddev=0.01, dtype=tf.float64)
        
        self.num_inputs = num_inputs
        self.num_outputs =  num_outputs
        
        self.input_layer = tf.placeholder(shape=[None, self.num_inputs],
                           dtype=tf.float64)
                           
                           
        self.layer1 = tf.layers.dense(self.inputs, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer1_size, activation=tf.nn.relu)
        
        self.layer2 = tf.layers.dense(self.inputs, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer1_size, activation=tf.nn.relu)
        
        self.layer3 = tf.layers.dense(self.inputs, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer1_size, activation=tf.nn.relu)
        
        self.layer4 = tf.layers.dense(self.inputs, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer1_size, activation=tf.nn.relu)
        
        self.output_layer = tf.layers.dense(self.inputs, 
                kernel_initializer= self._gauss_init, units=self.layer1_size)
        
        self.predicted_q_val = np.argmax(self.output_layer, 1)
        
        self.target_q = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        
        self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target_q, 
                                                        self.predicted_q_val)))

        self.update_model = tf.train.AdamOptimizer(self.learning_rate).\
        minimize(self.loss)
        
        
        
    def predict(self, inputs):
        return self.sess.run(self.predicted_q_val, 
        feed_dict = {self.input_layer:inputs})
    
    def train(self, inputs, target_q):
        pself.sess.run(self.update_model, feed_dict={self.input_layer:inputs,
                                self.target_q:target_q})





class ActionSpace:
    def __init__(self):
        self._action_list = [(hfo.DASH, 33.3, -90.0), (hfo.DASH, 66.6, -90.0), 
                    (hfo.DASH, 100.0, -90.0), (hfo.DASH, 30.0, 90.0), 
                    (hfo.DASH, 66.6, 90.0), (hfo.DASH, 1000, 90.0), 
                    tuple(hfo.NOOP)]
        
    def sample(self):
        return random.choice(self._action_list)
        
    def __getitem__(self, pos):
        return self._action_list[pos]
    
    def __len__(self):
        return len(self._action_list)
        
    def __setitem__(self. item, pos)
        self._action_list[pos] = item
    
    def __str__(self):
        return "list of actions for AI goalkeeper"

class StateSpace:
    def __init__(self):
        pass
        
        
        
class Goalie:

    def _connect_to_server(self):
        self.env.connectToServer(LOW_LEVEL_FEATURE_SET,
        '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', 
        6000, 'localhost', 'base_right', True)

    def __init__(self, state_space, action_space):
        self.state_space = state_space
        self.action_space = action_space
        self.env = hfo.HFOEnvironment()
        self._connect_to_server()
        
        
        
    def reset(self):
        pass
    
    def step(self):
        pass
    
    
        
        
class FixedGoalieExperiment(self):
    def __init__(self, learning_rate, num_episodes, update_freq,
        pre_train_stage, buffer_size):
        
        self.goalie = Goalie()
        self.learning_rate = learning_rate
        self.num_episodes = num_episodes
        self.update_freq = update_freq
        self.pre_train_stage = pre_train_stage
        self.buffer_size = buffer_size
    
    def run(self):
        pass
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    def log(self):
        pass 
       
   
        
        
        
        
        
