from agent import Agent
import numpy as np
import tensorflow as tf



# as feature set the high/low level feature set will be used

# first attempt will be to just use discrete actions without added 
# the continous actions that are available

class DeepQAgent:
    def __init__(self):
        pass
        
        
    def build_network(self, vec_len):
        input_layer = tf.placeholder(shape=[vec_len, 1], dtype=tf.float32)
        l1 = 0
        l2 = 0
        l3 = 0
        l4 = 0
