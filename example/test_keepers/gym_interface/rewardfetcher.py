import hfo
import helper
import math
class RewardFetcher:
    
    
    def __init__(self):
        self.norm = math.sqrt((68)**2+(52.5)**2)
        
    def reward(self, state, status):
        if status == hfo.GOAL:
            return -500
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
        elif status == hfo.OUT_OF_BOUNDS:
            return 250
        
        else:
            dist_norm = 1-(helper.euclid_dist(state[3:5], state[0:2])/self.norm)
            r = (state[9] * dist_norm)*100
            return r
            #return (state[9])*10
    
    
    
