import hfo
import random
class ActionSpace:
                
    def __init__(self, action_set):
        self._action_set = action_set
        self._action_list = []
        
    def __len__(self):
        return len(self._action_list)
    
    def sample(self):
        return self._action_list.index(random.choice(self._action_list))
        
    def __getitem__(self, pos):
        return self._action_list[pos]
    
    def __str__(self):
        return "action set" %self._action_set
