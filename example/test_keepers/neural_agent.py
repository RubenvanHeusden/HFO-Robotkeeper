import itertools
import math
from hfo import *
import argparse
from array import array
import numpy as np
import tensorflow as tf




def main():
    parser = argparse.ArgumentParser()
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=6000, type=int)
    parser.add_argument('--feature_set', default=HIGH_LEVEL_FEATURE_SET, type=int)
    parser.add_argument('--trials', default=10, type=int)
    args = parser.parse_args()
  
    hfo = HFOEnvironment()
    hfo.connectToServer(args.feature_set,
                      '/home/student/Desktop/HFO-master_ruben/bin/teams/base/config/formations-dt', args.port,
                      'localhost', 'base_right', True)
                      
    num_features = hfo.getStateSize()
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        status = IN_GAME
        for i in range(args.trials):
            
            while status == IN_GAME:
                features = hfo.getState()
                # run
                # do action
                # get reward
                
    
    




if __name__ == '__main__':
    main()
