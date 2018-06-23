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
import sys

def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)



def one_hot_vector(size, index):
    vec = np.zeros((1, size))
    vec[:, index] = 1
    return vec




def angle_to_ball(ball_pos, goalie_pos):
    return math.degrees(math.atan2(ball_pos[1]-goalie_pos[1],ball_pos[0]-goalie_pos[0]))

def distance_to_ball(ball_pos, goalie_pos):
    HALF_FIELD_WIDTH = 68 
    HALF_FIELD_LENGTH = 52.5 
    return math.sqrt((ball_pos[0] - goalie_pos[0])**2 + (ball_pos[1] - goalie_pos[1])**2) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)


def dist_goalie_to_ball_line(ball, end_point, goalie):
    return np.linalg.norm(np.cross(end_point-ball, ball-goalie))/np.linalg.norm(end_point-ball)

def advanced_reward_func(old_state, new_state):
    HALF_FIELD_WIDTH = 68 
    HALF_FIELD_LENGTH = 52.5 
    end_x = -0.83
    old_ball_pos = old_state[3:5]
    ball_pos = new_state[3:5]
    gradient = (ball_pos[1]-old_ball_pos[1]) / (ball_pos[0]-old_ball_pos[0])
    x_diff = end_x-ball_pos[0]
    y_diff = gradient*x_diff
    estimate_pos =  (end_x, ball_pos[1]+y_diff)
    old_dist =  dist_goalie_to_ball_line(old_ball_pos, estimate_pos, old_state[0:2])
    new_dist =  dist_goalie_to_ball_line(ball_pos, estimate_pos, new_state[0:2])
    #print (old_dist-new_dist) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)
    return (old_dist-new_dist) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)*100



# de extra functie berekend ook de advanced reward, maar alleen het punt op de 
# lijn langs de y-as


def advanced_reward_func_extra(old_state, new_state):
    HALF_FIELD_WIDTH = 68 
    HALF_FIELD_LENGTH = 52.5 
    end_x = -0.83
    old_ball_pos = old_state[3:5]
    ball_pos = new_state[3:5]
    gradient = (ball_pos[1]-old_ball_pos[1]) / (ball_pos[0]-old_ball_pos[0])
    x_diff = end_x-ball_pos[0]
    y_diff = gradient*x_diff
    estimate_pos =  (end_x, ball_pos[1]+y_diff)
    goal_x = old_state[0:2][0]
    
    x_diff2 = abs(goal_x-ball_pos[0])
    y_diff2 = gradient*x_diff2
    estimate_pos2 = (goal_x, ball_pos[1]+y_diff2) 
    
    
    
    
    
    
    
    old_dist =  distance_to_ball_y(estimate_pos2, old_state[0:2])
    new_dist =  distance_to_ball_y(estimate_pos2, new_state[0:2])
    
    return (old_dist-new_dist) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)*100







def distance_to_ball_y(ball_pos, goalie_pos):
    return math.sqrt((ball_pos[1] - goalie_pos[1])**2)



def bin_states(features, num_a=10, num_d=10):
    
    # angles = 5, distances = 10
    """ 
    params: number of bins, coordinates of point to be binned
    returns: bin in which the point 'pos' lies 
    
    """
    
    
    num_angles = num_a
    num_distances = num_d 
    
    angle_bins = np.array(np.linspace(-120, 120, num=num_angles))
    distance_bins = np.array(np.linspace(0, 130, num=num_distances))
    
    dist = distance_to_ball(features[3:5], features[0:2])
    angle = angle_to_ball(features[3:5], features[0:2])
    d_bin =  np.digitize(dist, distance_bins) 
    a_bin = np.digitize(angle, angle_bins)
    return a_bin*(num_distances-1)+d_bin
    
    
    
def bin_states_adv(features, num_angles=10, num_distances=3):
    distribution = [9, 7, 5]
    lin = []
    max_dist = 130
    for x in range(0, num_distances):
        left_bound = (max_dist/num_distances) * (x)
        right_bound = (max_dist/num_distances) * (x+1)
        lin.append(np.array(np.linspace(left_bound, right_bound, 
        num=distribution[x])))
    
    distance_bins =  np.unique(np.concatenate(lin))
    angle_bins = np.array(np.linspace(-120, 120, num=num_angles))
    dist = distance_to_ball(features[3:5], features[0:2])
    angle = angle_to_ball(features[3:5], features[0:2])
    d_bin =  np.digitize(dist, distance_bins) 
    a_bin = np.digitize(angle, angle_bins)
    
    return a_bin*(distance_bins.shape[0]-1)+d_bin   
    

    
    
    
def daan_method(old_state, new_state):
    HALF_FIELD_WIDTH = 68 
    HALF_FIELD_LENGTH = 52.5 
    old_ball_pos = old_state[3:5]
    old_player_pos = old_state[0:2]

    ball_pos = new_state[3:5]
    player_pos = new_state[0:2]
    

    old_dist = distance_to_ball_y(old_ball_pos, old_player_pos)
    new_dist = distance_to_ball_y(ball_pos, player_pos)

    return (old_dist-new_dist) * math.sqrt(HALF_FIELD_WIDTH**2 + HALF_FIELD_LENGTH**2)*100


def get_reward(state, status, prev_state=None):

        line_dist = advanced_reward_func(prev_state, state)
        
        if status == hfo.GOAL:
            return -500
            
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
            
        elif status == hfo.OUT_OF_TIME:
            return -500
        
        else:
            return line_dist


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
    
        self.layer1_size = 10
        self.layer2_size = 10
        self.layer3_size = 10
        self.layer4_size = 10


        self.learning_rate = learning_rate
        
        self._gauss_init = tf.truncated_normal_initializer(mean=0.0, 
                                            stddev=0.01, dtype=tf.float32)
        
        self.num_inputs = num_inputs
        self.num_outputs =  num_outputs
        
        self.input_layer = tf.placeholder(shape=[None, self.num_inputs],
                           dtype=tf.float32, name="input_layer")
                           
        with tf.name_scope("hidden_layer1"):
            self.layer1 = tf.contrib.layers.fully_connected(self.input_layer,
                                        num_outputs = self.layer1_size,
                                        weights_initializer=tf.contrib.layers.xavier_initializer())

        
        
                     
        with tf.name_scope("hidden_layer2"):
            self.layer2 = tf.contrib.layers.fully_connected(self.layer1, 
                                        num_outputs = self.layer2_size, 
                                        weights_initializer= tf.contrib.layers.xavier_initializer())

        with tf.name_scope("hidden_layer3"):       
            self.layer3 = tf.contrib.layers.fully_connected(self.layer2, 
                                        num_outputs = self.layer3_size, 
                                        weights_initializer= tf.contrib.layers.xavier_initializer())
                                        

        with tf.name_scope("hidden_layer4"):       
            self.layer4 = tf.contrib.layers.fully_connected(self.layer3, 
                                        num_outputs = self.layer4_size, 
                                        weights_initializer= tf.contrib.layers.xavier_initializer())
                                        
                                        

        with tf.name_scope("fully_connected_output_layer"):   
            self.output_layer = tf.contrib.layers.fully_connected(self.layer4,
                    num_outputs = num_outputs,  
                    weights_initializer = tf.contrib.layers.xavier_initializer(),activation_fn=None)

            


        
        with tf.name_scope("prediction"):
            self.predicted_q_val = tf.argmax(self.output_layer, 1)
        
        self.target_q = tf.placeholder(shape=[None, num_outputs], 
        dtype=tf.float32, name="target")
        
        with tf.name_scope("loss_function"):
            
            self.loss = tf.reduce_sum(tf.square(tf.subtract(self.target_q, 
                                                        self.output_layer)))
            
        with tf.name_scope("train_step"):
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
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
                    (hfo.DASH, 100.0, -90.0), (hfo.DASH, 33.3, 90.0), 
                    (hfo.DASH, 66.6, 90.0), (hfo.DASH, 100.0, 90.0)]# tuple([hfo.NOOP])]
        
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
        self._num_distances = num_distances   
    
    def state(self, features):
        s = bin_states(features, self._num_angles, self._num_distances)
        #s = bin_states_adv(features)
        return s
        
        
        
        
        
        
class Goalie:

    def _connect_to_server(self):
        self.env.connectToServer(hfo.HIGH_LEVEL_FEATURE_SET,
        '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', 
        6000, 'localhost', 'base_right', True)

    def __init__(self):
        self.state_space = StateSpace(10, 10)
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
        
        return s1, reward, done, (status == hfo.GOAL)
    
        
        
class FixedGoalieExperiment:
    def __init__(self, learning_rate, num_episodes, update_freq,
        pre_train_stage, buffer_size):
        
        self.tau = 0.001
        self._total_steps = 0
        self.goalie = Goalie()
        self.learning_rate = learning_rate
        self.e = 0.1
        self.gamma = 0.99
        self.batch_size = 16
        self._total_train_eps = 0
        self.num_episodes = num_episodes
        self.update_freq = update_freq
        self.pre_train_stage = pre_train_stage
        self.buffer_size = buffer_size
        self.q_network = NeuralNetwork(self.goalie.state_space.n, 
                                        self.goalie.action_space.n, self.learning_rate)
                                        
        self.target_network = NeuralNetwork(self.goalie.state_space.n, 
                                        self.goalie.action_space.n, self.learning_rate)
        

    def run(self, file_name):

        with tf.Session() as sess:
            exp_buffer = Memory(self.buffer_size)    

            init = tf.global_variables_initializer()
            saver = tf.train.Saver()
            trainables = tf.trainable_variables()

            targetOps = updateTargetGraph(trainables,self.tau)

            sess.run(init)
            captures = 0
            for x in xrange(self.num_episodes):
                s = self.goalie.reset()
                done = False
                action = [0]
                for y in xrange(500):
                
                                 
                    if np.random.rand(1) < self.e or  self._total_steps < self.pre_train_stage:
                        action[0] = self.goalie.action_space.sample()          
                    else:
                        action, all_q = sess.run([self.q_network.predicted_q_val,
                                 self.q_network.output_layer],
                                 feed_dict={self.q_network.input_layer:one_hot_vector(self.goalie.state_space.n, s)}) 
                    s1, reward, done, goal = self.goalie.step(action[0]) 
                    exp_buffer.add([s, action[0], reward, s1, done])                   
                    
                    if self._total_steps >= self.pre_train_stage:
                        if self._total_steps % self.update_freq == 0:
                            print "TRAINING !!!"
                            
                            update_batch = exp_buffer.sample(self.batch_size)
                            next_states = np.array([one_hot_vector(self.goalie.state_space.n, item[3]) for item in update_batch])
                            curr_states = np.array([one_hot_vector(self.goalie.state_space.n, item[0]) for item in update_batch])
                            best_actions = np.array([item[1] for item in update_batch])
                            rewards = np.array([item[2] for item in update_batch])
                             
                            maxq1, all_new_states = sess.run([self.q_network.predicted_q_val,self.q_network.output_layer], feed_dict = {self.q_network.input_layer:np.vstack(next_states)})  
                            target_q = all_new_states
                            for row in range(target_q.shape[0]):
                                target_q[row, best_actions[row]] = rewards[row] + self.gamma*maxq1[row]
                            _, l = sess.run([self.q_network.update_model, self.q_network.loss],
                            feed_dict={self.q_network.input_layer:np.vstack(curr_states), self.q_network.target_q:target_q})       
                            updateTarget(targetOps,sess)
                            self._total_train_eps+=1
                    self._total_steps+=1 
                    if done:
                        if goal:
                            captures = 0
                        else:
                            captures+=1
                            if captures == 100:
                                saver.save(sess, file_name)                               
                                print "perfect agent !!!"
                                sys.exit()
                        self.e = 1./((x/50) + 10)
                        break        
                    s = s1
                
   
                    
            saver.save(sess, file_name)
            
        
        
    
    def test(self, filename):
        saver = tf.train.Saver()
        with tf.Session() as sess:
            saver.restore(sess,filename)            
            for x in xrange(self.num_episodes):
                s = self.goalie.reset()
                done = False
                for y in xrange(200):
                    action, all_ = sess.run([self.q_network.predicted_q_val, self.q_network.output_layer], 
                                 feed_dict={self.q_network.input_layer:
                                 one_hot_vector(self.goalie.state_space.n, s)}) 

 
                    #if np.random.rand(1) < self.e:
                    #    action[0] = self.goalie.action_space.sample()          
                    s1, reward, done, _ = self.goalie.step(action[0])
                    s = s1        
                    if done:
                        break













    def load_model(self, f_name_org, f_name_new):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            saver.restore(sess, f_name_org)
            exp_buffer = Memory(self.buffer_size)    
            trainables = tf.trainable_variables()
            targetOps = updateTargetGraph(trainables,self.tau)

            for x in xrange(self.num_episodes):
                s = self.goalie.reset()
                done = False
                action = [0]  
                for y in xrange(500):
                
                    if np.random.rand(1) < self.e : #or self._total_steps < self.pre_train_stage
                        action[0] = self.goalie.action_space.sample()          
                    else:
                        action, all_q = sess.run([self.q_network.predicted_q_val,self.q_network.output_layer],
                                 feed_dict={self.q_network.input_layer:one_hot_vector(self.goalie.state_space.n, s)}) 
                    s1, reward, done, _ = self.goalie.step(action[0]) 
                    exp_buffer.add([s, action[0], reward, s1, done])                   
                    if self._total_steps >= self.pre_train_stage:
                        if self._total_steps % self.update_freq == 0:
                            print "TRAINING !!!"
                            
                            update_batch = exp_buffer.sample(self.batch_size)
                            next_states = np.array([one_hot_vector(self.goalie.state_space.n, item[3]) for item in update_batch])
                            curr_states = np.array([one_hot_vector(self.goalie.state_space.n, item[0]) for item in update_batch])
                            best_actions = np.array([item[1] for item in update_batch])
                            rewards = np.array([item[2] for item in update_batch])
                             
                            maxq1, all_new_states = sess.run([self.q_network.predicted_q_val,self.q_network.output_layer], feed_dict = {self.q_network.input_layer:np.vstack(next_states)})  
                            target_q = all_new_states
                            for row in range(target_q.shape[0]):
                                target_q[row, best_actions[row]] = rewards[row] + self.gamma*maxq1[row]
                            _, l = sess.run([self.q_network.update_model, self.q_network.loss],
                            feed_dict={self.q_network.input_layer:np.vstack(curr_states), self.q_network.target_q:target_q})       
                            updateTarget(targetOps,sess)
                            self._total_train_eps+=1
                    self._total_steps+=1 
                    if done:
                        # changed e !
                        self.e = 1./((x/50) + 10)
                        break        
                    s = s1
                                    

                    
            saver.save(sess, f_name_new)                
    


        
def main(f_name, num_trials):
    obj = FixedGoalieExperiment(learning_rate=0.0001, num_episodes=num_trials, update_freq=200,
        pre_train_stage=4000, buffer_size=500000)
    
    obj.run(f_name)
    #obj.test(f_name)
    #obj.load_model(f_name, f_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='models/model1', type=str)
    parser.add_argument('--trials', default=500, type=int)
    args=parser.parse_args()
    tf.reset_default_graph()
    main(args.filename, args.trials)
    
    
    
    
