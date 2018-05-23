from collections import deque
import random
import numpy as np

class Memory:
	def __init__(self, size):
		self._data = []
		self.size = size
		
	def add(self, item):
		if len(self._data) >= self.size:
			self._data.pop(0)
		self._data.append(item)
		
		
	def get_batch(self, batch_size=10):
		return random.sample(self._data, k=batch_size)
		
	def __getitem__(self, pos):
		return self._data[pos]
			
	def __len__(self):
		return len(self._data)
