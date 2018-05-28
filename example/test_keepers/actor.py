import numpy as np
import tensorflow as tf

from gym_interface.lowlevelrandomagent import LowLevelRandomAgent

class Actor:
    def __init__(self, sess, num_inputs, num_actions, num_params):
        
        self.sess = sess
        self.batch_size = 32
        self.l1_size = 256
        self.l2_size = 128
        self.l3_size = 64
        self.l4_size = 32
        self.action_bound = 100
        self._kernel_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float64)
        self.learning_rate = 0.0001
        self.action_bound = 1
        
        #self.scaled_out = tf.multiply(out, self.action_bound)
        
        self.input_layer = tf.placeholder(shape=[None, num_inputs], dtype=tf.float64)
        
        self._layer1 = tf.layers.dense(inputs = self.input_layer, kernel_initializer = self._kernel_init, activation=tf.nn.relu, units = self.l1_size)
        self._layer2 = tf.layers.dense(inputs = self._layer1, kernel_initializer = self._kernel_init, activation=tf.nn.relu, units = self.l2_size)
        self._layer3 = tf.layers.dense(inputs = self._layer2, kernel_initializer = self._kernel_init, activation=tf.nn.relu, units = self.l3_size)
        self._layer4 = tf.layers.dense(inputs = self._layer3, kernel_initializer = self._kernel_init, activation=tf.nn.relu, units = self.l4_size)
        
        self.act_out = tf.layers.dense(inputs = self._layer4, kernel_initializer = self._kernel_init, units = num_actions)
        self.param_out = tf.layers.dense(inputs = self._layer4, kernel_initializer = self._kernel_init, units = num_params)
        
        
        

        
        self.network_params = tf.trainable_variables()
        self.action_gradient = tf.placeholder(shape = [None, num_params], dtype = tf.float64) 
        
        self.unnormalized_actor_gradients = tf.gradients(
        self.param_out, self.network_params, - self.action_gradient)
        
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

        # Optimization Op
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).\
        apply_gradients(zip(self.actor_gradients, self.network_params))
    
    def train(self, inputs, action_gradient):
        self.sess.run(self.optimize, feed_dict = {self.input_layer:inputs, self.action_gradient:action_gradient})
        
    def predict(self, inputs):
        return self.sess.run(_, feed_dict={self.input_layer:inputs})
    
    def predict_target_network(self):
        pass
        
    def update_target_network(self):
        pass
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
