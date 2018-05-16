from statespace import StateSpace
import numpy as np 
import helper
import math

class OneHotStateSpace(StateSpace):
    
    def __init__(self, num_states):
        StateSpace.__init__(self, num_states)
        self._num_states = num_states
        
    def get_state(self, features):
        vec = np.zeros((self._num_states, 1))
        size = math.sqrt(self._num_states)
        pos = helper.bin_ball_position(features[3:5],[-1, 1], [-1, 1],size, size)
        vec[pos] = 1
        
        return vec
