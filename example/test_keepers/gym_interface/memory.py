import random

class Memory:
    def __init__(self):
        self._contents = []
        
    def __len__(self):
        return len(self._contents)
    
    def __getitem__(self, pos):
        return self._contents[pos]
    
    def __setitem__(self, pos, item):
        if type(item) != tuple || len(item) != 4:
            raise ValueError
        self._contents[pos] = item
        
    def shuffle(self):
        random.shuffle(self._contents)
        
