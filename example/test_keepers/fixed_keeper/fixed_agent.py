# This is an attempt to create a keeper agent that is stationary on the 
# goal line and just stops a single straight through shot by moving 
# in a discrete action space


import random
import hfo
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt


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

    return d_bin*num_distances+a_bin
    

def get_reward(state, status):

        if status == hfo.GOAL:
            return -500
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
        elif status == hfo.OUT_OF_BOUNDS:
            return 250
        
        else:
            
            return int((1-abs(state[51]))*10)
            


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
                           dtype=tf.float64, name="input_layer")
                           
        with tf.name_scope("hidden_layer1"):
            self.layer1 = tf.contrib.layers.fully_connected(self.input_layer,biases_initializer=None,
                                        num_outputs = self.layer1_size,
                                        weights_initializer= self._gauss_init)
        with tf.variable_scope("fully_connected", reuse=True):        
            tf.summary.histogram('layer1_weights', tf.get_variable('weights', dtype=tf.float64))
     

            
                     
                            
        with tf.name_scope("hidden_layer2"):
            self.layer2 = tf.contrib.layers.fully_connected(self.layer1, biases_initializer=None,
                                        num_outputs = self.layer2_size, 
                                        weights_initializer= self._gauss_init)
        with tf.variable_scope("fully_connected_1", reuse=True):        
            tf.summary.histogram('layer2_weights', tf.get_variable('weights', dtype=tf.float64))
         


 
        with tf.name_scope("hidden_layer3"):       
            self.layer3 = tf.contrib.layers.fully_connected(self.layer2, biases_initializer=None,
                                        num_outputs = self.layer3_size, 
                                        weights_initializer= self._gauss_init)
                                        
        with tf.variable_scope("fully_connected_2", reuse=True):        
            tf.summary.histogram('layer3_weights', tf.get_variable('weights', dtype=tf.float64))
       

        with tf.name_scope("hidden_layer4"):
            self.layer4 = tf.contrib.layers.fully_connected(self.layer3, biases_initializer=None,
                                        num_outputs = self.layer4_size,  
                                        weights_initializer= self._gauss_init)
                                        
        with tf.variable_scope("fully_connected_3", reuse=True):        
            tf.summary.histogram('layer4_weights', tf.get_variable('weights', dtype=tf.float64))
      

        with tf.name_scope("fully_connected_output_layer"):   
            self.output_layer = tf.contrib.layers.fully_connected(self.layer4,biases_initializer=None,
                    num_outputs = num_outputs,  
                    weights_initializer = self._gauss_init, 
                    activation_fn=None)
        tf.summary.histogram('Q_Values', self.output_layer)         
        with tf.variable_scope("fully_connected_4", reuse=True):        
            tf.summary.histogram('output_layer_weights', tf.get_variable('weights', dtype=tf.float64))
            


        
        with tf.name_scope("prediction"):
            self.predicted_q_val = tf.argmax(self.output_layer, 1)
        
        self.target_q = tf.placeholder(shape=[None, num_outputs], 
        dtype=tf.float64, name="target")
        
        with tf.name_scope("loss_function"):
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target_q, 
                                                        self.output_layer)))
            tf.summary.scalar('loss', self.loss)
            
        with tf.name_scope("train_step"):
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
        done = int(not(status == hfo.IN_GAME))
        
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


    def run(self):

        with tf.Session() as sess:
            exp_buffer = Memory(self.buffer_size)    
            merged_summary = tf.summary.merge_all()
            init = tf.global_variables_initializer()
            sess.run(init)
            writer = tf.summary.FileWriter("stats/test20")
            writer.add_graph(sess.graph)
            
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
                          print "XXXXXXXXXXXXXXXXXXX TRIAINING !!! XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX"
                          train_batch = exp_buffer.sample(self.batch_size)
                          
                          states = np.vstack(np.array([item[0] for item in train_batch]))
                          best_actions = np.vstack(np.array([item[1] for item in train_batch]))
                          rewards = np.vstack(np.array([item[2] for item in train_batch]))
                          next_states = np.vstack(np.array([item[3] for item in train_batch]))
                          
                          #train_batch = tf.nn.batch_normalization(tf.convert_to_tensor(train_batch), mean, variance, offset=None, scale=None)
                             
                             
                          maxq1 = sess.run(self.q_network.predicted_q_val, feed_dict = {self.q_network.input_layer:next_states})
                          vals = rewards + self.gamma*maxq1
                          target_q = all_q
                          print vals[0]
                          for z in range(all_q.shape[0]):
                              target_q[z, best_actions[z][0]] = vals[z]
                           
                          data = sess.run(merged_summary,
                          feed_dict={self.q_network.input_layer:states, self.q_network.target_q:target_q})  
                          writer.add_summary(data, self._total_train_eps)
                                                     
                          _, l = sess.run([self.q_network.update_model, self.q_network.loss],
                          feed_dict={self.q_network.input_layer:states, self.q_network.target_q:target_q})       

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
        
   

   
        
        
        
        
        
        
        
        
        
def main():
    FixedGoalieExperiment(learning_rate=0.01, num_episodes=100, update_freq=500,
        pre_train_stage=1000, buffer_size=1000).run()


if __name__ == "__main__":
    main()












