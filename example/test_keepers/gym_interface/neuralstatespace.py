from statespace import StateSpace
import numpy as np

class NeuralStateSpace(StateSpace):
    def __init__(self):
        StateSpace.__init__(self)
    
    def get_state(self, features):
        flat_f = features.flatten()
        result = flat_f[np.newaxis, :]
        return result
