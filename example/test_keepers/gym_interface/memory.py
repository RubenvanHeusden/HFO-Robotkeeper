from collections import deque
import random
import numpy

class Memory:
	def __init__(self, size):
		self._data = deque(maxlen=size)


	def __len__(self):
		return len(self._data)

	def add(self, item):
		"""default is to add to the end of the queue"""
		if len(item) != 4 :
			raise ValueError("the size of the item is " +str(len(item))+", should be 4")
		self._data.extend(item)
		return None
	
	def get_batch(self, batch_size=10):
		if batch_size > len(self._data):
			raise ValueError("batch size "+str(batch_size)+" exceeds total size of deque : "+str(len(self._data)))
		return random.sample(self._data, k=batch_size)	

	def __str__(self):
		return str(self._data)

	def __repr__(self):
		return str(self._data)
