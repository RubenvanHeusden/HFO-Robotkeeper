import hfo
import helper
class RewardFetcher:
    
    
    def __init__(self):
        pass
    
    def reward(self, state, status):
        if status == hfo.GOAL:
            return -500
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
        elif status == hfo.OUT_OF_BOUNDS:
            return 250
        
        else:
            r = state[9] * helper.euclid_dist(state[3:5], state[0:2])*100
            return r
            #return (state[9])*10
    
    
    
