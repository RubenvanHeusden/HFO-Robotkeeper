import tensorflow as tf
import numpy as np
from gym_interface.memory import Memory
from gym_interface.lowlevelrandomagent import LowLevelRandomAgent

#trainables = tf.trainable_variables()

# set the environment
tf.reset_default_graph()
env = LowLevelRandomAgent()

e = 0.1
gamma = 0.99
num_episodes = 100000
update_freq = 50

update_freq = 1000
init_steps = 500
total_steps = 0


# setting parameters
num_features = env.getStateSize()
num_outputs = len(env.action_set)
learning_rate = 0.0001
layer1_size = 64
layer2_size = 32
layer3_size = 16
batch_size = 32
experience_buffer = Memory(size=5000)

# We use this distribution to set the weights for our hidden layers
gauss_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01, dtype=tf.float32)

# The input vector contains the features that will be used  for
# action prediction


inputs = tf.placeholder(shape = [None, num_features],dtype=tf.float32)
layer1 = tf.layers.dense(inputs, kernel_initializer= gauss_init,
                         units=layer1_size, activation=tf.nn.relu)
                         
layer2 = tf.layers.dense(layer1, kernel_initializer = gauss_init,
                         units=layer2_size, activation=tf.nn.relu)
                         
layer3 = tf.layers.dense(layer2, kernel_initializer = gauss_init,
                         units =layer3_size, activation=tf.nn.relu)
                         
outputs = tf.layers.dense(layer3, kernel_initializer = gauss_init, 
                        units=num_outputs)

t_q = tf.placeholder(shape = [None, num_outputs], dtype=tf.float32)

predict = tf.argmax(outputs, 1)

loss = tf.reduce_sum(tf.square(t_q - outputs))

update = tf.train.AdamOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for i in range(num_episodes):
        s = env.reset()
        episode_buffer = Memory(100)
        done = False
        for j in range(100):

            if np.random.rand(1) < e:
                action = env.action_set.sample()
            else:
            	action = sess.run(predict, feed_dict = {inputs:s})[0]
            s1, reward, done = env.step(action)

            
            episode_buffer.add([s, action, reward, s1, done])
            
            # check if we have filled our buffer with some experience
            if total_steps > init_steps:
				if total_steps % update_freq == 0:
				    print "updating !!!"
				    exp_batch = np.array(experience_buffer.get_batch(batch_size)).reshape((batch_size, 5))
				    
				    q_vals = sess.run(outputs, feed_dict = {inputs:np.vstack(exp_batch[:, 0])})
				    _, maxq1 = sess.run([outputs, predict], feed_dict = {inputs:np.vstack(exp_batch[:, 3])})
				    vals = exp_batch[:, 2] + gamma*maxq1
				    target_q = q_vals
				    for x in range(q_vals.shape[0]):
				    	target_q[x, exp_batch[:, 1][x]] = vals[x]
				    
				    _ = sess.run(update,feed_dict={inputs:np.vstack(exp_batch[:, 0]), t_q:target_q})
	    
            s = s1
            if done:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
            total_steps+=1
        
        for item in episode_buffer._data:
        	experience_buffer.add(item)    
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
    
















