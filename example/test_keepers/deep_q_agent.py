import tensorflow as tf
import numpy as np
from gym_interface.memory import Memory
tf.reset_default_graph()
from gym_interface.lowlevelrandomagent import LowLevelRandomAgent

#trainables = tf.trainable_variables()

# set the environment

env = LowLevelRandomAgent()

e = 0.1
gamma = 0.99
num_episodes = 100
# setting parameters
num_features = env.getStateSize()
num_outputs = len(env.action_set)
learning_rate = 0.0001
layer1_size = 32
layer2_size = 16
layer3_size = 8
batch_size = 32
experience = Memory(size=5000)

# We use this distribution to set the weights for our hidden layers
gauss_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

# The input vector contains the features that will be used  for
# action prediction


inputs = tf.placeholder(shape = [None, num_features],dtype=tf.float32)
layer1 = tf.layers.dense(inputs, kernel_initializer= gauss_init,
                         units=layer1_size, activation=tf.nn.relu)
                         
layer2 = tf.layers.dense(inputs=layer1, kernel_initializer = gauss_init,
                         units=layer2_size, activation=tf.nn.relu)
                         
layer3 = tf.layers.dense(inputs=layer2, kernel_initializer = gauss_init,
                         units =layer3_size, activation=tf.nn.relu)
                         
outputs = tf.layers.dense(inputs=layer3, kernel_initializer = gauss_init, 
                        units=num_outputs)

t_q = tf.placeholder(shape = [1, num_outputs], dtype=tf.float32)

predict = tf.argmax(outputs, 1)

loss = tf.reduce_sum(tf.square(t_q - outputs))

update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.getState().reshape((1, num_features))
        done = False
        for j in range(100):
            action, all_q = sess.run([predict, outputs], feed_dict = {inputs:s})
            
            if np.random.rand(1) < e:
                action[0] = env.action_set.sample()
            
            
            s1, reward, done = env.step(action[0])
            s1 = s1.reshape((1, num_features))
            q1 = sess.run(outputs, feed_dict = {inputs:s1})
            
            max_q1 = np.max(q1)
            target_q = all_q
            target_q[0, action[0]] = reward + gamma*max_q1
            
            _ = sess.run(update,feed_dict={inputs:s,t_q:target_q})
            
            s = s1
            if done:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
















