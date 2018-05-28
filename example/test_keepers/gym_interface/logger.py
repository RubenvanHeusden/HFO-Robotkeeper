import numpy as np

class Logger:
    def __init__(self, f_name):
        self.f_name = f_name
        self.data = []
		
		
    def log(self, num_stops):
        self.data.append(num_stops)

    def on_exit(self):
        arr = np.array(self.data)
        arr.tofile(self.f_name)
		
	
	
