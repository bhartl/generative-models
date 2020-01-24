import os
import numpy as np
import ruamel.yaml as yaml


def first_moment(data):
	return data


def second_moment(data):
	# np.apply_along_axis(lambda xi: np.outer(xi, xi).flatten(), axis=1, arr=x)
	return np.concatenate(data[None, ...]*data.T[..., None], axis=1)


class MaximumEntropyModel(object):

	def __init__(self, features=first_moment, initial_weights=None, data=None, l1=0., l2=0., file=None):
		self._features = features
		self._weights = np.asarray(initial_weights) if initial_weights is not None else None
		self._l1 = l1
		self._l2 = l2
		self._data = data

		self._positive_phase = None     # <features_i>_data
		self._positive_negative = None  # <features_i>_model

		self._batch_size = None

		self._data_batch = None
		self._model_batch = None
		self._max_steps = None

		self._step = 0
		self._file = file
		self.history = self.load(file)

	def features(self, data):
		data = np.asarray(data)
		assert data.ndim == 2

		try:
			features = self._features(data)

		except:
			features = []  # np.empty((len(data), len(self._weights)))
			for i, foo in enumerate(self._features):
				features.append(foo(data))

			features = np.ascontiguousarray(np.concatenate(features, axis=1))

		return features

	def __init_weights(self):

		bs, self._batch_size = self._batch_size, 2

		try:
			f = self.features(self.load_minibatch())
			self._weights = np.random.rand(f.shape[1])/f.shape[1]

		finally:
			self._batch_size = bs

		return self._weights

	def negative_energy(self, data):
		features = self.features(data)

		if np.ndim(features) == 2:
			return np.apply_along_axis(lambda x: x.dot(self._weights), axis=1, arr=features)

		return features.dot(self._weights)

	def partition_function(self, samples):
		negative_energy = self.negative_energy(samples)
		return np.average(np.exp(negative_energy))

	def cost(self, data, samples):
		avg_data_energy = -np.average(self.negative_energy(data))
		batch_partition_function = -np.log(self.partition_function(samples))

		cost_reg = 0.
		if self._l1 != 0:
			cost_reg += self._l1 * np.linalg.norm(self._weights, ord=1)

		if self._l2 != 0:
			cost_reg += self._l2 * np.linalg.norm(self._weights, ord=2)

		return avg_data_energy - batch_partition_function + cost_reg

	def gradient(self, data, samples):
		"""

		:param data:
		:param samples: fantasy particles
		:return:
		"""

		self._positive_phase = -np.average(self.features(data), axis=0)
		self._negative_phase = -np.average(self.features(samples), axis=0)

		grad_reg = 0.
		if self._l1 != 0:
			norm_w = np.absolute(self._weights)
			grad_reg = self._l1 * 2 * self._weights / norm_w
			grad_reg[norm_w == 0.] = 0.

		if self._l2 != 0:
			grad_reg = self._l2 * 2 * np.absolute(self._weights)

		return self._positive_phase - self._negative_phase + grad_reg

	def sample(self, **kwargs):
		raise NotImplementedError('sample')

	def warm_up(self, **sample_kwargs):
		bs, self._batch_size = self._batch_size, 1

		try:
			if self._weights is None:
				self.__init_weights()

			print('*** warm up ***')
			s = self.sample(**sample_kwargs)
			print('')
		except:
			raise
		finally:
			self._batch_size=bs

		return s[0]

	def monitor(self, end='\n', **kwargs):
		try:
			cost = self.history['cost'][-1][-1]
		except:
			cost = '---'

		print('\rstep: {0}/{1}, cost: {2:.3f}'.format(self._step, self._max_steps, cost), end=end)

	def load_minibatch(self, batch_size=None):
		self_batch = False
		if batch_size is None:
			batch_size = self._batch_size
			self_batch = True

		data_batch = np.random.choice(np.arange(len(self._data)), size=batch_size, replace=False)
		data_batch = np.ascontiguousarray(self._data[data_batch])

		if self_batch:
			self._data_batch = data_batch

		return data_batch

	def fit(self, data=None, batch_size=10, max_steps=1000, learning_rate=1e-2, **sample_kwargs):

		print('*** start fitting maxent model ***')

		if data is not None:
			self._data = data

		if self._weights is None:
			self.__init_weights()

		self._batch_size = batch_size
		self._max_steps = max_steps + self._step

		cost = []
		self.history['cost'].append(cost)
		self._weights = np.ascontiguousarray(self._weights)
		self.history['weights'].append(self._weights)

		for _ in range(self._step, self._max_steps):
			self._step += 1
			self._data_batch = self.load_minibatch()
			self._model_batch = self.sample(**sample_kwargs)

			gradient = self.gradient(data=self._data_batch, samples=self._model_batch)
			self._weights -= gradient * learning_rate

			cost.append(self.cost(data=self._data_batch, samples=self._model_batch))
			self.monitor(end='')
			self.dump(path=self._file)

		self.monitor(end='\n')
		self.dump(path=self._file)

		print('** dumped results to `{}` ***'.format(self._file))
		print('** done ***')

		return self.history

	def load(self, path):
		try:
			assert path is not None

			with open(path, 'r') as f:
				history = yaml.load(f)

			for i, wi in enumerate(history['weights']):
				history['weights'][i] = np.ascontiguousarray(wi)

			self._step = int(np.sum([len(c) for c in history['cost']]))
			self._weights = history['weights'][-1] if len(history['weights']) > 0 else self._weights

		except Exception:
			history = dict(cost=[], weights=[])
			self._step = 0

		self.history = history
		return self.history

	def dump(self, path):
		h = dict(cost=[], weights=[])
		h['cost'] = [np.asarray(c).tolist() for c in self.history['cost']]
		h['weights'] = [np.asarray(w).tolist() for w in self.history['weights']]

		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		with open(path, 'w') as f:
			yaml.safe_dump(h, f)
