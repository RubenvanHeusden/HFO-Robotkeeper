import matplotlib.pyplot as plt
import numpy as np

class Visualizer:
	def __init__(self, f_path='stats.bin'):
		self.f_path = f_path
		self._data = np.fromfile(f_path)
		
	def display(self,save=False):

		x = np.array(range(self._data.shape[0]))
		
		plt.plot(x, self._data)
		plt.ylim(0, self._data.shape[0])
		plt.show()
