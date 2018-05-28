import argparse
import hfo
import tensorflow as tf
import numpy as np
from gym_interface.lowlevelrandomagent import LowLevelRandomAgent
from actor import Actor
from critic import Critic
from gym_interface.memory import Memory

def run(num_episodes, buf_size):
    # initialize the tensorflow variables

    e = 0.1
    
    # initializing the HFO environment
    env = LowLevelRandomAgent()
    state_space_dim = env.getStateSize()
    action_space_dim = env.action_space.n
    
    
    # initializing the experienc_replay buffer
    exp_buf = Memory(buf_size)
    
    print env.action_space.params
    
    with tf.Session() as sess:
        
        # initializing the actor and the critic networks
        actor = Actor(sess, num_inputs = state_space_dim, 
                                num_actions=action_space_dim, 
                                num_params=env.action_space.params)        
        
        critic = Critic(sess, num_inputs=100, num_outputs=1)

        # calling the variable initter AFTER the init of the networks
        init = tf.global_variables_initializer()
        sess.run(init)

        
        
        # setting the initial variables for the HFO environments
        for x in range(num_episodes):
            total_reward = 0
            s = env.reset()
            done = False
            
            for x in range(100):
                #TODO : random action according to boltzmann thin in Caitlin's paper
                #TODO : something with random noise (see code about DDPG)
                actions = actor.predict(s)
                s1, reward, done = env.step(actions[0])
                
            s = s1    
                
        


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', default=1000, type=int)
    parser.add_argument('--port', default=6000, type=int)
    parser.add_argument('--buffer_size', default=10000, type=int)
    args = parser.parse_args()
    
    # run the main program
    run(num_episodes=args.trials,buf_size=args.buffer_size)
    

if __name__ == "__main__":
    main()
