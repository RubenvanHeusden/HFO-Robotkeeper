import hfo
import helper
import math
class RewardFetcher:
    
    
    def __init__(self):
        #self.norm = math.sqrt((68)**2+(52.5)**2)
        pass
        
    def reward(self, state, status):
        if status == hfo.GOAL:
            return -500
        elif status == hfo.CAPTURED_BY_DEFENSE:
            return 500
        elif status == hfo.OUT_OF_BOUNDS:
            return 250
        
        else:
            sin, cos = state[51:53]
            return (cos*10-0.5) + (1-abs(sin)-0.5)*10 
            # is now a row vector !!!
            #return (state[9])*10
    

    
