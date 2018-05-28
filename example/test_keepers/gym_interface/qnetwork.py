import tensorflow as tf
import numpy as np

class QNetwork:
    def __init__(self, input_size, output_size):
        
        self.layer1_size = 256
        self.layer2_size = 128
        self.layer3_size = 64
        self.layer4_size = 32
        
        self.learning_rate = 0.000000001
        
        self._gauss_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float64)


        self.inputs = tf.placeholder(shape = [None, input_size],dtype=tf.float64)
       
        self.layer1 = tf.layers.dense(self.inputs, kernel_initializer= self._gauss_init,
                         units=self.layer1_size, activation=tf.nn.relu)
        
        self.layer2 = tf.layers.dense(self.layer1, kernel_initializer = self._gauss_init,
                         units=self.layer2_size, activation=tf.nn.relu)
        
        self.layer3 = tf.layers.dense(self.layer2, kernel_initializer = self._gauss_init,
                         units =self.layer3_size, activation=tf.nn.relu)

        self.layer4 = tf.layers.dense(self.layer3, kernel_initializer = self._gauss_init,
                         units =self.layer4_size, activation=tf.nn.relu)
        
        self.outputs = tf.layers.dense(self.layer4, kernel_initializer = self._gauss_init, 
                        units=output_size)
        
        
        self.t_q = tf.placeholder(shape = [None, output_size], dtype=tf.float64)

        self.predict = tf.argmax(self.outputs, 1)

        self.loss = tf.reduce_sum(tf.square(tf.subtract(self.t_q, self.outputs)))

        self.update = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

