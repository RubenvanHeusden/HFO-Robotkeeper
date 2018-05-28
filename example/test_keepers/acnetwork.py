import argparse
import hfo
import tensorflow as tf
import numpy as np
from gym_interface.lowlevelrandomagent import LowLevelRandomAgent
from actor import Actor
from critic import Critic


def run():
    #env = LowLevelRandomAgent()
    #action_space_dim = env.action_space.n
    action_bound = 0
    

    
    with tf.Session() as sess:
        actor = Actor(sess, num_inputs = 10, num_actions=3, num_params=6)        

def main():
    run()
    

if __name__ == "__main__":
    main()
