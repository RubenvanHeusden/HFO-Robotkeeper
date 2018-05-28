from __future__ import division
import tensorflow as tf
import numpy as np
from gym_interface.memory import Memory
from gym_interface.lowlevelrandomagent import LowLevelRandomAgent
from gym_interface.visualizer import Visualizer
from gym_interface.qnetwork import QNetwork

import time
import matplotlib.pyplot as plt
#trainables = tf.trainable_variables()
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

# set the environment
tf.reset_default_graph()
env = LowLevelRandomAgent()

gamma = 0.99 # was 0.99 
num_episodes = 10000
update_freq = 5000
init_steps = 2000
total_steps = 0
train_cnt = 0
C = 20
losses = []


start_e = 1
end_e = 0.1
ann_steps = 10000

e = start_e
stepDrop = (start_e - end_e)/ann_steps


# setting parameters
num_features = env.getStateSize()
num_outputs = len(env.action_space)
batch_size = 128
experience_buffer = Memory(size=10000)


# We use this distribution to set the weights for our hidden layers
qnet1 = QNetwork(num_features, num_outputs)
qnet2 = QNetwork(num_features, num_outputs)
v = Visualizer()
# The input vector contains the features that will be used  for
# action prediction

init = tf.global_variables_initializer()
tau = 0.001
trainables = tf.trainable_variables()
targetOps = updateTargetGraph(trainables, tau)


with tf.Session() as sess:
    sess.run(init)
    for i in xrange(num_episodes):
        s = env.reset()
        episode_buffer = Memory(100)
        done = False
        j = 0
        while j < 99:
            j+=1
            if np.random.rand(1) < e:
                action = env.action_space.sample()
                #print "DOING RANDOM STUFF"
            else:
                action = sess.run(qnet1.predict, feed_dict = {qnet1.inputs:s})[0]
            s1, reward, done = env.step(action)

            
            episode_buffer.add([s, action, reward, s1, done])
            
            # check if we have filled our buffer with some experience
            if total_steps > init_steps:
                if total_steps % update_freq == 0:
                    if e > end_e:
                        e -= stepDrop
                    print "updating !!!"
                    if train_cnt % C == 0:
                        updateTarget(targetOps,sess)
                    
                    exp_batch = np.array(experience_buffer.get_batch(batch_size)).reshape((batch_size, 5))
                    
                    q_vals = sess.run(qnet2.outputs, feed_dict = {qnet2.inputs:np.vstack(exp_batch[:, 0])})
                    _, maxq1 = sess.run([qnet2.outputs, qnet2.predict], feed_dict = {qnet2.inputs:np.vstack(exp_batch[:, 3])})
                    vals = exp_batch[:, 2] + gamma*maxq1
                    target_q = q_vals
                    for x in range(q_vals.shape[0]):
                        target_q[x, exp_batch[:, 1][x]] = vals[x]
                    
                    _, l = sess.run([qnet1.update, qnet1.loss],feed_dict={qnet1.inputs:np.vstack(exp_batch[:, 0]), qnet1.t_q:target_q})    
                    losses.append(l)
                    train_cnt+=1
            s = s1
            if done:
                #Reduce chance of random action as we train the model.
                break
            total_steps+=1
        
        for item in episode_buffer._data:
             experience_buffer.add(item)    
    env.logger.on_exit()    
    time.sleep(5)
    print train_cnt, C
    v.display()      
    print losses
    plt.plot(losses)
    plt.show()
            
            
            
            
            
            
            
            
            
            
            
            
            
    
















