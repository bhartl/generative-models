# generative-models

An implementation of the Maximum Entropy Generative model based as discussed in https://arxiv.org/abs/1803.08823 can be found in `gempy.maximum_entropy_model.py`

In `gempy.mnist.mnist_generator.py` we apply the Maximum Entropy Principle to generate (or fit) a Ising-based generative model on single numbers in the mnist-dataset. `gempy.mnist.mnist_generator.ipynb` provides a classifyer cnn as well as `MNISTGenerator` instances which can be trained to generate numbers from `0` to `9` using the Maximum Entropy Principle, which can then be evaluated with the classifyer network.

