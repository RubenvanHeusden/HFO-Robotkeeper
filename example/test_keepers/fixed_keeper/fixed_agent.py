# This is an attempt to create a keeper agent that is stationary on the 
# goal line and just stops a single straight through shot by moving 
# in a discrete action space


import random
import hfo
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)


def bin_states(features):
    """ 
    params: number of bins, coordinates of point to be binned
    returns: bin in which the point 'pos' lies 
    
    """
    num_angles = 20
    num_distances = 5 
    
    angle_bins = np.array(np.linspace(-90, 90, num=num_angles))
    distance_bins = np.array(np.linspace(0, 1, num=num_distances))
    
    dist = features[53]
    angle = math.degrees(math.asin(features[51]))
    
    d_bin =  np.digitize(dist, distance_bins) 
    a_bin = np.digitize(angle, angle_bins)

    return d_bin*num_distances+num_angles
    

def get_reward(state, status):

        if status == hfo.GOAL:
            return -500
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
        elif status == hfo.OUT_OF_BOUNDS:
            return 250
        
        else:
            
            return (1-abs(state[51]))*10
            


class Memory:
    def __init__(self, max_size):
        self._data = []
        self.max_size = max_size
    
    def add(self, item):
        if len(self._data) >= self.max_size:
            self._data.pop(0)
        self._data.append(item)
        return None
        
    def sample(self, size):
        return random.sample(self._data, k=size)

    def __len__(self):
        return len(self._data)
    
    def __setitem__(self, item, pos):
        self._data[pos] = item
    
    def __getitem__(self, pos):
        return self._data[pos]



# Neural network used for the fixed line keeper experiment
# Vanilla, DQN, DOUBLE Q, DUELLING Q
class NeuralNetwork:
    
    def __init__(self, num_inputs, num_outputs):
    
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
                           
                           
        self.layer1 = tf.layers.dense(self.input_layer, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer1_size, activation=tf.nn.relu)
        
        self.layer2 = tf.layers.dense(self.layer1, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer2_size, activation=tf.nn.relu)
        
        self.layer3 = tf.layers.dense(self.layer2, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer3_size, activation=tf.nn.relu)
        
        self.layer4 = tf.layers.dense(self.layer3, 
                                    kernel_initializer= self._gauss_init,
                         units=self.layer4_size, activation=tf.nn.relu)
        
        self.output_layer = tf.layers.dense(self.layer4, 
                kernel_initializer= self._gauss_init, units=num_outputs)
        
        self.predicted_q_val = tf.argmax(self.output_layer, 1)
        
        self.target_q = tf.placeholder(shape=[None, num_outputs], dtype=tf.float64)
        
        self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target_q, 
                                                        self.output_layer)))

        self.update_model = tf.train.AdamOptimizer(self.learning_rate).\
        minimize(self.loss)
        
        
        
    def predict(self, sess, inputs):
        return sess.run(self.predicted_q_val, 
        feed_dict = {self.input_layer:inputs})
    
    def train(self, sess, inputs, target_q):
        sess.run(self.update_model, feed_dict={self.input_layer:inputs,
                                self.target_q:target_q})





class ActionSpace:
    def __init__(self):
        self._action_list = [(hfo.DASH, 33.3, -90.0), (hfo.DASH, 66.6, -90.0), 
                    (hfo.DASH, 100.0, -90.0), (hfo.DASH, 30.0, 90.0), 
                    (hfo.DASH, 66.6, 90.0), (hfo.DASH, 1000, 90.0), 
                    tuple([hfo.NOOP])]
        
        self.n = len(self._action_list)
        
    def sample(self):
        return self._action_list.index(random.choice(self._action_list))
        
    def __getitem__(self, pos):
        return self._action_list[pos]
    
    def __len__(self):
        return len(self._action_list)
        
    def __setitem__(self, item, pos):
        self._action_list[pos] = item
    
    def __str__(self):
        return "list of actions for AI goalkeeper"
        
        
        

class StateSpace:
    def __init__(self, num_angles, num_distances):
        self.n = num_angles*num_distances
    
    def state(self, features):
        s = bin_states(features)
        return np.identity(self.n)[s:s+1]
        
        
        
        
        
        
class Goalie:

    def _connect_to_server(self):
        self.env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
        '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', 
        6000, 'localhost', 'base_right', True)

    def __init__(self):
        self.state_space = StateSpace(10, 20)
        self.action_space = ActionSpace()
        self.env = hfo.HFOEnvironment()
        self._connect_to_server()
        
        
        
    def reset(self):
        features = self.env.getState()
        return self.state_space.state(features)
    
    def step(self, action):
        # we need a new state, a reward and a 'done'
        self.env.act(*self.action_space[action])
        status = self.env.step()
        features = self.env.getState()
        s1 = self.state_space.state(features)
        reward = get_reward(features, status)
        done = not(status == hfo.IN_GAME)
        
        return s1, reward, done
    
        
        
class FixedGoalieExperiment:
    def __init__(self, learning_rate, num_episodes, update_freq,
        pre_train_stage, buffer_size):
        
        self.losses = []
        self._total_steps = 0
        self.goalie = Goalie()
        self.learning_rate = learning_rate
        self.e = 0.1
        self.gamma = 0.99
        self.batch_size = 64
        self._total_train_eps = 0 
        self.num_episodes = num_episodes
        self.update_freq = update_freq
        self.pre_train_stage = pre_train_stage
        self.buffer_size = buffer_size
        self.q_network = NeuralNetwork(self.goalie.state_space.n, 
                                        self.goalie.action_space.n)

        self.q_network2 = NeuralNetwork(self.goalie.state_space.n, 
                                        self.goalie.action_space.n)


        
        self.trainables = tf.trainable_variables()
        self.targetOps = updateTargetGraph(self.trainables,0.001)
        self.C = 3

    def run(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            exp_buffer = Memory(self.buffer_size)    
            for x in xrange(self.num_episodes):
                s = self.goalie.reset()
                episode_buffer = Memory(100)
                done = False
                for y in xrange(100):
                    action, all_q = sess.run([self.q_network.predicted_q_val,
                                 self.q_network.output_layer],
                                 feed_dict={self.q_network.input_layer:
                                 s}) 
 
 
                    if np.random.rand(1) < self.e:
                        action[0] = self.goalie.action_space.sample()
                                        
                    s1, reward, done = self.goalie.step(action[0])
                    episode_buffer.add([s, action, reward, s1, done])
                    
                    if self._total_steps > self.pre_train_stage:
                       if self._total_steps % self.update_freq == 0:
                            if self._total_train_eps % self.C == 0:
                                updateTarget(self.targetOps,sess)
                          
                            print "XXXXXXXXXXXXXXXXXXX TRIAINING !!! XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                            train_batch = np.array(exp_buffer.sample(self.batch_size)).reshape((self.batch_size, 5))                    
                            q_vals = sess.run(self.q_network2.output_layer, feed_dict = {self.q_network2.input_layer:np.vstack(train_batch[:, 0])})
                            _, maxq1 = sess.run([self.q_network2.output_layer, self.q_network2.predicted_q_val], feed_dict = {self.q_network2.input_layer:np.vstack(train_batch[:, 3])})
                            vals = train_batch[:, 2] + self.gamma*maxq1
                            target_q = q_vals
                            for x in range(q_vals.shape[0]):
                                target_q[x, train_batch[:, 1][x]] = vals[x]
                    
                            _, l = sess.run([self.q_network.update_model, self.q_network.loss],feed_dict={self.q_network.input_layer:np.vstack(train_batch[:, 0]), self.q_network.target_q:target_q})            
                            self.losses.append(l)              
                            self._total_train_eps+=1
                    if done:
                        self.e = 1./((x/50) + 10)
                        break        
                    s = s1
                
                    self._total_steps+=1
                for item in episode_buffer:
                    exp_buffer.add(item)            
            plt.plot(self.losses)
            plt.show()       
        
    
    def log(self):
        """ something with TensorBoard"""
        pass 
       
   
        
        
        
        
        
        
        
        
        
        
a = FixedGoalieExperiment(learning_rate=0.1, num_episodes=1000, update_freq=500,
        pre_train_stage=1000, buffer_size=1000)

a.run()
















