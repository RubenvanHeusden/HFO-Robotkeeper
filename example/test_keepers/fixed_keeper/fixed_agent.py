# This is an attempt to create a keeper agent that is stationary on the 
# goal line and just stops a single straight through shot by moving 
# in a discrete action space
from __future__ import division
import random
import hfo
import numpy as np
import tensorflow as tf
import math
import matplotlib.pyplot as plt
import argparse




def angle_to_ball(ball_pos, goalie_pos):
    return math.degrees(math.atan2(ball_pos[1]-goalie_pos[1],ball_pos[0]-goalie_pos[0]))

def distance_to_ball(ball_pos, goalie_pos):
    HALF_FIELD_WIDTH = 68 
    HALF_FIELD_LENGTH = 52.5 
    return math.sqrt((ball_pos[0] - goalie_pos[0])**2 + (ball_pos[1] - goalie_pos[1])**2) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)


def dist_goalie_to_ball_line(ball, end_point, goalie):
    return np.linalg.norm(np.cross(end_point-ball, ball-goalie))/np.linalg.norm(end_point-ball)

def advanced_reward_func(old_state, new_state):
    end_x = -0.83
    old_ball_pos = old_state[3:5]
    ball_pos = new_state[3:5]
    gradient = (ball_pos[1]-old_ball_pos[1]) / (ball_pos[0]-old_ball_pos[0])
    x_diff = end_x-ball_pos[0]
    y_diff = gradient*x_diff
    estimate_pos =  (end_x, ball_pos[1]+y_diff)
    result =  dist_goalie_to_ball_line(ball_pos, estimate_pos, new_state[0:2])
    return result









def bin_states(features, num_angles, num_distances):
    """ 
    params: number of bins, coordinates of point to be binned
    returns: bin in which the point 'pos' lies 
    
    """
    
    angle_bins = np.array(np.linspace(-90, 90, num=num_angles))
    distance_bins = np.array(np.linspace(0, 1, num=num_distances))
    
    dist = features[53]
    angle = math.degrees(math.asin(features[51]))
    
    d_bin =  np.digitize(dist, distance_bins) 
    a_bin = np.digitize(angle, angle_bins)

    return d_bin*num_distances+a_bin




def get_reward(state, status, prev_state=None):
        #HALF_FIELD_WIDTH = 68 
        #HALF_FIELD_LENGTH = 52.5 

        #line_dist = advanced_reward_func(prev_state, state) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)
        
        if status == hfo.GOAL:
            return -500
            
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
        elif status == hfo.OUT_OF_BOUNDS:
            return 250
        
        else:
        
            #print ((((0-abs(state[51]))*10)+1) * (1+state[53]))
            
            
            
            
            #norm = abs(state[53]) 
            
            #dist_offset = (state[53]-extra)*70 *norm
            #angle = ((((0-abs(state[51]))*10)+1)) *(1-norm)
            #print angle+dist_offset
            #print dist_offset*3+angle
            #return dist_offset*3+angle
            #dist_weight = (110-angle_to_ball(state[3:5], state[0:2]))/20
            #return 35-abs(angle_to_ball(state[3:5], state[0:2]))*dist_weight
            print (((0-abs(state[51]))*10)+1)
            return  (((0-abs(state[51]))*10)+1)

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
    
    def __init__(self, num_inputs, num_outputs, learning_rate):
    
        self.layer1_size = 512
        self.layer2_size = 256
        self.layer3_size = 128
        self.layer4_size = 64

        self.learning_rate = learning_rate
        
        self._gauss_init = tf.truncated_normal_initializer(mean=0.0, 
                                            stddev=0.01, dtype=tf.float64)
        
        self.num_inputs = num_inputs
        self.num_outputs =  num_outputs
        
        self.input_layer = tf.placeholder(shape=[None, self.num_inputs],
                           dtype=tf.float64, name="input_layer")
                           
        with tf.name_scope("hidden_layer1"):
            self.layer1 = tf.contrib.layers.fully_connected(self.input_layer,
                                        num_outputs = self.layer1_size,
                                        weights_initializer= self._gauss_init)

                            
        with tf.name_scope("hidden_layer2"):
            self.layer2 = tf.contrib.layers.fully_connected(self.layer1, 
                                        num_outputs = self.layer2_size, 
                                        weights_initializer= self._gauss_init)

         


 
        with tf.name_scope("hidden_layer3"):       
            self.layer3 = tf.contrib.layers.fully_connected(self.layer2, 
                                        num_outputs = self.layer3_size, 
                                        weights_initializer= self._gauss_init)
                                        

       

        with tf.name_scope("hidden_layer4"):
            self.layer4 = tf.contrib.layers.fully_connected(self.layer3, 
                                        num_outputs = self.layer4_size,  
                                        weights_initializer= self._gauss_init)
                                        

      

        with tf.name_scope("fully_connected_output_layer"):   
            self.output_layer = tf.contrib.layers.fully_connected(self.layer4,
                    num_outputs = num_outputs,  
                    weights_initializer = self._gauss_init, 
                    activation_fn=None)

            


        
        with tf.name_scope("prediction"):
            self.predicted_q_val = tf.argmax(self.output_layer, 1)
        
        self.target_q = tf.placeholder(shape=[None, num_outputs], 
        dtype=tf.float64, name="target")
        
        with tf.name_scope("loss_function"):
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target_q, 
                                                        self.output_layer)))
            
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
    
        #self._action_list = [(hfo.DASH, i, -90.0) for i in range(0, 120, 20)]+[(hfo.DASH, j, 90.0) for j in range(0, 120, 20)] 
        self._action_list = [(hfo.DASH, 33.3, -90.0), (hfo.DASH, 66.6, -90.0), 
                    (hfo.DASH, 100.0, -90.0), (hfo.DASH, 30.0, 90.0), 
                    (hfo.DASH, 66.6, 90.0), (hfo.DASH, 100.0, 90.0)]
        
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
        self._num_angles = num_angles
        self._num_distances=num_distances   
    
    def state(self, features):
        s = bin_states(features, self._num_angles, self._num_distances)
        return np.identity(self.n)[s:s+1]
        
        
        
        
        
        
class Goalie:

    def _connect_to_server(self):
        self.env.connectToServer(hfo.LOW_LEVEL_FEATURE_SET,
        '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', 
        6000, 'localhost', 'base_right', True)

    def __init__(self):
        self.state_space = StateSpace(20, 10)
        self.action_space = ActionSpace()
        self.env = hfo.HFOEnvironment()
        self._connect_to_server()
        
        
        
    def reset(self):
        features = self.env.getState()
        return self.state_space.state(features)
    
    def step(self, action):
        # we need a new state, a reward and a 'done'
        old_state = self.env.getState()
        self.env.act(*self.action_space[action])
        status = self.env.step()
        features = self.env.getState()
        s1 = self.state_space.state(features)
        reward = get_reward(features, status, old_state)
        done = int(not(status == hfo.IN_GAME))
        
        return s1, reward, done
    
        
        
class FixedGoalieExperiment:
    def __init__(self, learning_rate, num_episodes, update_freq,
        pre_train_stage, buffer_size):
        
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
                                        self.goalie.action_space.n, self.learning_rate)


    def run(self, file_name):

        with tf.Session() as sess:
            exp_buffer = Memory(self.buffer_size)    

            init = tf.global_variables_initializer()
            sess.run(init)
            saver = tf.train.Saver()
            
            for x in xrange(self.num_episodes):
                s = self.goalie.reset()
                done = False
                for y in xrange(100):
                    action, all_q = sess.run([self.q_network.predicted_q_val,
                                 self.q_network.output_layer],
                                 feed_dict={self.q_network.input_layer:
                                 s}) 

 
                    if np.random.rand(1) < self.e:
                        action[0] = self.goalie.action_space.sample()          
                    s1, reward, done = self.goalie.step(action[0])
                    #exp_buffer.add([s, action[0], reward, s1, done])                   
                    if self._total_steps >= self.pre_train_stage:
                        pass
                        #print exp_buffer.sample(64)
                          #train_batch = tf.nn.batch_normalization(tf.convert_to_tensor(train_batch), mean, variance, offset=None, scale=None)
                             
                             
                    maxq1, all_ = sess.run([self.q_network.predicted_q_val,self.q_network.output_layer], feed_dict = {self.q_network.input_layer:s1})
                    vals = reward + self.gamma*maxq1
                    target_q = all_q
                    target_q[0,action[0]] = reward + self.gamma*maxq1

                    _, l = sess.run([self.q_network.update_model, self.q_network.loss],
                    feed_dict={self.q_network.input_layer:s, self.q_network.target_q:target_q})       

                    self._total_train_eps+=1
                    if done:
                        self.e = 1./((x/50) + 10)
                        break        
                    s = s1
                
                    self._total_steps+=1    
                    
            saver.save(sess, file_name)
            
        
        
    
    def test(self, filename):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(filename+".meta")
            saver.restore(sess,tf.train.latest_checkpoint('./'))










            
            for x in xrange(self.num_episodes):
                s = self.goalie.reset()
                done = False
                for y in xrange(200):
                    action, all_ = sess.run([self.q_network.predicted_q_val, self.q_network.output_layer], 
                                 feed_dict={self.q_network.input_layer:
                                 s}) 

 
                    #if np.random.rand(1) < self.e:
                    #    action[0] = self.goalie.action_space.sample()          
                    s1, reward, done = self.goalie.step(action[0])
                    s = s1        
                    if done:
                        break

        
        
        
def main(f_name):
    obj = FixedGoalieExperiment(learning_rate=0.001, num_episodes=5000, update_freq=500,
        pre_train_stage=1000, buffer_size=5000)
    
    obj.run(f_name)
    #obj.test(f_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='models/model1', type=str)
    args=parser.parse_args()
    tf.reset_default_graph()
    main(args.filename)












