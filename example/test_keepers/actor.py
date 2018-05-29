import numpy as np
import tensorflow as tf

from gym_interface.lowlevelrandomagent import LowLevelRandomAgent

class Actor:
    def __init__(self, sess, num_inputs, num_actions, num_params):
        
        #TODO : add batch normalization
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
        
        self.min_bounds = tf.constant([[-100.0, -180.0, -180.0, -180.0]], dtype=tf.float64)
        self.max_bounds = tf.constant([[100.0, 180.0, 180.0, 180.0]], dtype=tf.float64)
        
        
        
        # the input layer for the actor network is a state S
        self.input_layer = tf.placeholder(shape=[None, num_inputs], dtype=tf.float64)
        
        self._layer1 = tf.layers.dense(inputs = self.input_layer, 
                    kernel_initializer = self._kernel_init, 
                    activation=tf.nn.relu, units = self.l1_size)
                    
        self._layer2 = tf.layers.dense(inputs = self._layer1, 
                    kernel_initializer = self._kernel_init, 
                    activation=tf.nn.relu, units = self.l2_size)
                    
        self._layer3 = tf.layers.dense(inputs = self._layer2, 
                    kernel_initializer = self._kernel_init, 
                    activation=tf.nn.relu, units = self.l3_size)
        
        self._layer4 = tf.layers.dense(inputs = self._layer3, 
                    kernel_initializer = self._kernel_init, 
                    activation=tf.nn.relu, units = self.l4_size)
        
        # The act_out layer outputs four discrete actions
        self.act_out = tf.layers.dense(inputs = self._layer4, 
                kernel_initializer = self._kernel_init, units = num_actions)
        
        # TODO : zeroing gradients

        
        
        # The param_out layer outputs the continious parameters for the discrete action
        self.param_out = tf.layers.dense(inputs = self._layer4, 
                kernel_initializer = self._kernel_init, units = num_params)
        
        #TODO : USE inverting gradients to let the output scale itself
        self.action_gradient = tf.placeholder(shape = [None, num_actions], dtype=tf.float64)
        
        # the variables of the network
        self.network_params = tf.trainable_variables()
        
        #TODO : BELOW CODE MIGHT BE WRONG
        self._unnormalized_actor_gradients = tf.gradients(
                    tf.concat([self.act_out, self.param_out], axis = 1), 
                    self.network_params, 
                    -self.action_gradient)
        
        # actor gradients  
        self._actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), 
                             self._unnormalized_actor_gradients))
        
        
        
        # calculating the loss for the actor network using the gradients
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).\
                apply_gradients(zip(self._actor_gradients, self.network_params))
        
        
        
        
    def predict(self, inputs):
        return self.sess.run([self.act_out, self.param_out], 
                    feed_dict={self.input_layer:inputs})
        
    def train(self, inputs, gradients):
        self.sess.run(self.optimizer, feed_dict = {self.input_layer:inputs,
                        self.action_gradients:gradients})
        
    def predict_all(self, inputs):
        return self.sess.run([self.act_predict, self.param_out], 
                    feed_dict={self.input_layer:inputs})            
        
        
        
        
        
        
        
        
