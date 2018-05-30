import numpy as np
import tensorflow as tf


class Critic:
    def __init__(self, sess, state_size, action_size, param_size, num_outputs):
    
        #TODO : add batch normalization
        self.l1_size = 256
        self.l2_size = 128
        self.l3_size = 64
        self.l4_size = 32
        self.learning_rate = 0.0001
        self.sess = sess
        # providing a probibility distribution for the initialization of 
        # the weights
        
        self._kernel_init = tf.truncated_normal_initializer(mean=0.0, 
                                                stddev=0.01, dtype=tf.float64)
        
        
        # input consists of 
        self.input_layer = tf.placeholder(shape=[None, state_size+action_size+param_size], dtype=tf.float64)
        
        self.state_input, self.action_input, self.param_input = tf.split(self.input_layer, [state_size, action_size, param_size], 1)
        
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
        
        self.out = tf.layers.dense(inputs = self._layer4, 
                    kernel_initializer = self._kernel_init, 
                    units = num_outputs)
                    
        
        
        # predicted Q value that comes from the target critic network
        self.predicted_q_value = tf.placeholder(shape=[None, 1], dtype=tf.float64)
        self.loss = np.sum(tf.square(tf.subtract(self.predicted_q_value, self.out)))
        
        self.update_model = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
        self.action_grads = tf.gradients(self.out, self.action_input)
        
    def predict(self, inputs):
        return self.sess.run(self.out, feed_dict = {self.input_layer:inputs})
        
        
    def train(self):
        pass 
    
    def get_gradients(self, inputs, state, action, params):
        return self.sess.run(self.action_input, 
        feed_dict = {self.input_layer:inputs})
        
        
        
        
        
        
        
        
        
        
        
        
        
        
