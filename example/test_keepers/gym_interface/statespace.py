import hfo
import helper
import numpy as np
import math
class StateSpace:
    def __init__(self, num_states):
        self._num_states = num_states
        
    def get_state(self, features):
        size = math.sqrt(self._num_states)
        pos = helper.bin_ball_position(features[3:5],[-1, 1], [-1, 1],size, size)
        return int(pos)
        
    def __len__(self):
        return self._num_states
        
