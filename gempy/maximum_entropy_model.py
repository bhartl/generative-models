import os
import numpy as np
import ruamel.yaml as yaml
import time


def first_moment(data):
	"""`first_moment` feature function

	:param data: data to evaluate feature function on
	:return: returns data as is
	"""
	return data


def second_moment(data):
	"""`second_moment` feature function

	:param data: data to evaluate feature function on
	:return: squared data
	"""
	return data**2


class MaximumEntropyModel(object):
	"""Maximum entropyGenerative model based as discussed in https://arxiv.org/abs/1803.08823"""

	def __init__(self, features=(first_moment, second_moment), initial_weights=None, data=None, l1=0., l2=0., file=None):
		"""Construct MaximumEntropyModel instance

		:param features: callable or list of callable representing features or list of features (will be concatenated)
		:param initial_weights: initial values for weights array, needs to be of same shape as concatenated features
		:param data: data to fit
		:param l1: l1 regularization strength, defaults to 0.
		:param l2: l2 regularization strength, defaults to 0.
		:param file: output file path for fit history (weights and losses, can be used to continue fit)
		"""
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
		""" return features evaluated on data

		:param data: samples of data  (2d array)
		:return: concatenated values of feature functions, evaluated on sample data
		"""
		data = np.asarray(data)
		assert data.ndim == 2

		try:  # if features is directly callable
			features = self._features(data)

		except:  # list of features, iteratively evaluate
			features = []
			for i, foo in enumerate(self._features):
				features.append(foo(data))

			features = np.ascontiguousarray(np.concatenate(features, axis=1))

		return features

	def __init_weights(self):
		"""initialize weights array knowing the shape of the features with random values scaled by number of features"""

		bs, self._batch_size = self._batch_size, 2

		try:
			f = self.features(self.load_minibatch())
			self._weights = np.random.rand(f.shape[1])/f.shape[1]

		finally:
			self._batch_size = bs

		return self._weights

	def negative_energy(self, data):
		"""evaluate negative value of energy on sample data (2d array), i.e. sum_i f_i lambda_i"""
		features = self.features(data)

		if np.ndim(features) == 2:
			return np.apply_along_axis(lambda x: x.dot(self._weights), axis=1, arr=features)

		return features.dot(self._weights)

	@property
	def loss(self):
		"""norm of gradient, i.e. sum_i|<f_i>_data - <f_i>_model|"""
		return np.linalg.norm(self._positive_phase - self._negative_phase)

	def gradient(self, data, samples):
		"""evaluate gradient of data and samples (including regularization)

		:param data: minibatch of data drawn from data-set
		:param samples: fantasy particles sampled from current parametrization of model
		:return: <f_i>_data - <f_i>_model - gradient_regularizers, where `i` runs over weights and features
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
		"""dummy routine, to be overwritten, which returns list of fantasy particles drawn from the current model"""
		raise NotImplementedError('sample')

	def warm_up(self, **sample_kwargs):
		"""initial execution of self.sample, number of samples set to 1

		:param sample_kwargs: passed to self.sample method
		:return: single sampled configuration
		"""
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
		"""monitor routine to be called during fitting"""
		try:
			loss = self.history['loss'][-1][-1]
		except:
			loss = '---'

		print('\rstep: {0}/{1}, loss: {2:.3f}'.format(self._step, self._max_steps, loss), end=end)

	def load_minibatch(self, batch_size=None):
		"""load minibatch of size batch_size (or self._batch_size if argument is None) from loaded data,
		if batch_size is None, minibatch is written to self._data_batch property"""
		self_batch = False
		if batch_size is None:
			batch_size = self._batch_size
			self_batch = True

		data_batch = np.random.choice(np.arange(len(self._data)), size=batch_size, replace=False)
		data_batch = np.ascontiguousarray(self._data[data_batch])

		if self_batch:
			self._data_batch = data_batch

		return data_batch

	def fit(self, data=None, batch_size=10, max_steps=1000, learning_rate=1e-2, dump_interval=10, **sample_kwargs):
		"""perform maximum entropy fitting on training data

		:param data: data to be fitted, defaults to None (need to be present in the this case)
		:param batch_size: mini-batch size for gradient evaluation
		:param max_steps: maximum number of fit steps
		:param learning_rate: learning rate in stochastic gradient descent step (non-addaptive atm)
		:param sample_kwargs: kwargs passed to self.sample routine
		:param dump_interval: interval for dumping history object
		:return: sample history object containing loss and weight data of fitting procedure
		"""

		print('*** start fitting maxent model ***')
		start = time.time()

		if data is not None:
			self._data = data

		if self._weights is None:
			self.__init_weights()

		self._batch_size = batch_size
		self._max_steps = max_steps + self._step

		loss = []
		self.history['loss'].append(loss)
		self._weights = np.ascontiguousarray(self._weights)
		self.history['weights'].append(self._weights)

		for i in range(self._step, self._max_steps):
			self._step += 1
			self._data_batch = self.load_minibatch()
			self._model_batch = self.sample(**sample_kwargs)

			gradient = self.gradient(data=self._data_batch, samples=self._model_batch)
			self._weights -= gradient * learning_rate

			loss.append(self.loss)
			self.monitor(end='')

			# dump history file each dump_interval steps
			if not np.mod(i, dump_interval):
				self.dump(path=self._file)

		self.monitor(end='\n')
		self.dump(path=self._file)

		print('** dumped results to `{}` ***'.format(self._file))
		print('** done after {} seconds ***'.format(time.time() - start))

		return self.history

	def load(self, path):
		"""load MaximumEntropyModel fitting history (weights and losses) to continue with previous run

		:param path: path to history file
		:return: history object (dictionary containing weights and losses)
		"""
		try:
			assert path is not None

			with open(path, 'r') as f:
				history = yaml.load(f)

			for i, wi in enumerate(history['weights']):
				history['weights'][i] = np.ascontiguousarray(wi)

			self._step = int(np.sum([len(c) for c in history['loss']]))
			self._weights = history['weights'][-1] if len(history['weights']) > 0 else self._weights

		except Exception:
			history = dict(loss=[], weights=[])
			self._step = 0

		self.history = history
		return self.history

	def dump(self, path):
		"""dump fitting history as yaml-file to specified path argument

		:param path: path for to-be-written yaml-file
		"""
		h = dict(loss=[], weights=[])
		h['loss'] = [np.asarray(c).tolist() for c in self.history['loss']]
		h['weights'] = [np.asarray(w).tolist() for w in self.history['weights']]

		directory = os.path.dirname(path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		with open(path, 'w') as f:
			yaml.safe_dump(h, f)
