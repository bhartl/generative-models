import matplotlib.pyplot as plt
from gempy import MaximumEntropyModel, first_moment
import numpy as np
from numba import jit
import os


def on_site(data):
	return data


@jit(nopython=True)
def nearest_neighbour(data, i=-1):

	if i == -1:
		features = np.empty((len(data), 28 * 28 * 4 - 28*4))
	else:
		features = np.empty((len(data), 4))

	i_features = 0
	for xi in range(len(data)):
		i_features = 0
		di = data[xi]

		if i > -1:
			sites = range(i, i+1)
		else:
			sites = range(len(data[xi]))

		for si in sites:
			row = si // 28

			left_neigh = si - 1
			right_neigh = si + 1
			top_neigh = si - 28
			bottom_neigh = si + 28

			if left_neigh >= row * 28:
				features[xi, i_features] = di[si] * di[left_neigh]
				i_features = i_features + 1

			if right_neigh < (row + 1) * 28:
				features[xi, i_features] = di[si] * di[right_neigh]
				i_features = i_features + 1

			if top_neigh > 0:
				features[xi, i_features] = di[si] * di[top_neigh]
				i_features = i_features + 1

			if bottom_neigh < len(data[xi]):
				features[xi, i_features] = di[si] * di[bottom_neigh]
				i_features = i_features + 1

	return features[:, :i_features]


def nearest_neighbour_mapping(x):

	features = []
	i_features = 0
	i_offset = 28*28

	for i in range(len(x)):
		features_i = []
		row = i // 28

		left_neigh = i - 1
		right_neigh = i + 1
		top_neigh = i - 28
		bottom_neigh = i + 28

		if left_neigh >= row * 28:
			features_i.append(i_features + i_offset)
			i_features = i_features + 1

		if right_neigh < (row + 1) * 28:
			features_i.append(i_features + i_offset)
			i_features = i_features + 1

		if top_neigh > 0:
			features_i.append(i_features + i_offset)
			i_features = i_features + 1

		if bottom_neigh < len(x):
			features_i.append(i_features + i_offset)
			i_features = i_features + 1

		features.append(np.ascontiguousarray(features_i))

	return features[:i_features]


@jit(nopython=True)
def delta_energy(x, weights, sweeped_x, sweeped_idx, neighs_idx):
	delta_energy = -(x[0, sweeped_idx] - sweeped_x) * weights[sweeped_idx]  # on site

	neighs = nearest_neighbour(x, i=sweeped_idx)[0]
	delta_energy -= np.sum((x[0, sweeped_idx] - sweeped_x) * neighs * weights[neighs_idx])

	return delta_energy


class MnistGenerator(MaximumEntropyModel):

	def __init__(self, **kwargs):
		MaximumEntropyModel.__init__(self, features=(on_site, nearest_neighbour), **kwargs)
		self._nearest_neighbour_mapping = None
		self._update_rate = 0.

	def sample(self, x0=None, n_sweeps=1, beta=10., save_fig=False):
		samples = []
		self._model_batch = samples

		if x0 is None:
			x0 = np.ascontiguousarray(np.random.rand(self.load_minibatch(1)[0].shape[0]))

		x = x0
		self._nearest_neighbour_mapping = nearest_neighbour_mapping(x)

		proposed, accepted = 0, 0
		while len(samples) < self._batch_size:
			energy_x = -self.negative_energy(x[None, :])
			proposed, accepted = 0, 0

			for n in range(n_sweeps):
				for m in range(len(x)):
					sweeped_x, sweeped_idx = self.sweep(x)
					# energy_new = -self.negative_energy(x[None, :])
					energy_new = energy_x + self.sweep_energy(x, sweeped_x, sweeped_idx)
					proposed += 1

					if self.detailed_balance(energy_x, energy_new, beta=beta):
						energy_x = energy_new
						accepted += 1
					else:
						x[sweeped_idx] = sweeped_x

			samples.append(np.copy(x))
			if not len(samples) % 4:
				self._update_rate = accepted / proposed
				self.monitor(end='', save_fig=False)

		self._update_rate = accepted / proposed
		self.monitor(end='', save_fig=save_fig if (isinstance(save_fig, bool) or not save_fig) else (not self._step % save_fig))
		self._model_batch = np.ascontiguousarray(samples)

		return self._model_batch

	def sweep(self, x):
		sweep_idx = np.random.randint(0, len(x))  # draw random integer from low (inclusive) to high (exclusive)

		prev_x, x[sweep_idx] = x[sweep_idx], np.random.rand()  # float(not(np.round(x[sweep_idx])))  #

		return prev_x, sweep_idx

	def sweep_energy(self, x, sweeped_x, sweeped_idx):
		neighs_idx = self._nearest_neighbour_mapping[sweeped_idx]
		return delta_energy(x[None, :], self._weights, sweeped_x, sweeped_idx, neighs_idx)

	@staticmethod
	def detailed_balance(energy_old, energy_new, beta=10.):
		r = np.exp(-(energy_new-energy_old)*beta)
		return np.random.rand() < min(1., r)

	def monitor(self, end='\n', save_fig=False):
		try:
			cost = '{:.3f}'.format(self.history['cost'][-1][-1])
		except:
			cost = '---'

		print(
			'\rstep: {0}/{1}, drawn samples: {2} ({3:.3f}), cost: {4}'.format(
				self._step, self._max_steps, len(self._model_batch), self._update_rate, cost
			),
			end=end
		)

		if save_fig:
			f, axes = plt.subplots(1, 2, sharex=True, sharey=True)

			ax1, ax2 = axes

			ax1.imshow(self._model_batch[0].reshape((28, 28)), cmap='binary')
			ax1.set_aspect('equal', 'box')
			ax1.set_title('sample')

			c2 = ax2.imshow(self._weights[:len(self._model_batch[0])].reshape(28, 28), vmin=-1., vmax=1., cmap='coolwarm')
			ax2.set_aspect('equal', 'box')
			ax2.set_title('weights (on-site)')

			f.colorbar(c2, ax=axes.ravel().tolist())

			# plt.tight_layout()
			plt.savefig(self._file.replace('.yml', '') + '-sample_{:04d}.png'.format(self._step))
			plt.close()

	@classmethod
	def main(cls, train_on=2, maxsteps=1000, batch_size=16, n_sweeps=1, learning_rate=1e-3, save_fig: (bool, int)=0, reg_l1=0., reg_l2=0.07, path='dat/mnist/'):
		import warnings
		warnings.simplefilter("ignore")

		from keras.datasets import mnist

		# input image dimensions
		img_rows, img_cols = 28, 28

		# load mnist data set
		(X_train, y_train), (X_test, y_test) = mnist.load_data()

		# flatten input (for maxent search)
		X_train = X_train.reshape(X_train.shape[0], img_rows * img_cols)
		X_test = X_test.reshape(X_test.shape[0], img_rows * img_cols)

		# Normalizing data
		X_train = X_train.astype("float32")
		X_test = X_test.astype("float32")
		X_train /= 255.0
		X_test /= 255.0

		train_path = os.path.join(path, 'number_{}.yml'.format(train_on))
		X_train = np.concatenate((X_train[y_train == train_on], X_test[y_test == train_on]))
		print('*** train on {} `{}`s in the mnist dataset ***'.format(len(X_train), train_on))

		mnist = cls(
			data=X_train,
			file=train_path,
			l2=reg_l2,
			l1=reg_l1,
		)

		# x0 = mnist.warm_up(n_sweeps=500, beta=100., x0=np.ascontiguousarray(np.random.rand(X_train.shape[1])))
		x0 = np.ascontiguousarray(np.random.rand(X_train.shape[1]))

		mnist.fit(
			max_steps=maxsteps,
			batch_size=batch_size,
			learning_rate=learning_rate,
			x0=x0,
			n_sweeps=n_sweeps,
			# beta=100.,
			save_fig=save_fig,
		)

		plt.figure()
		for i, c in enumerate(mnist.history['cost']):
			plt.plot(c, label='run ' + str(i))
		plt.xlabel('steps')
		plt.ylabel('cost')
		plt.legend()
		plt.show()


if __name__ == '__main__':
	import argh
	argh.dispatch_command(MnistGenerator.main)