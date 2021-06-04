from __future__ import annotations
from numpy import ndarray, ndim, product
import torch.nn as nn
import torch.cuda
import torch.tensor
import torch.nn.functional as F


class Encoder(nn.Module):
    """ pytorch based encoder: x -> z """

    def __init__(self, latent_labels: (str, tuple, list, set, None) = 'z', **kwargs):
        """
        :param latent_labels: Label or list of labels, specifying the number of outputs of the encoder (defaults to 'z').
                              This is important for the Variational Auto Encoder, where the Encoder must provide both,
                              a mean and a (log-) standard deviation value as output, i.e., two latent labels.

                              If latent_labels is None -> the latent space will be directly addressed as tensor/numpy
                              array, otherwise a it will be wrapped as a dictionary with keys proviced by latent_labels.

        """
        super(Encoder, self).__init__()

        self._latent_labels = None
        self.latent_labels = latent_labels

        self._latent = None
        self._latent_torch = None

        self.kwargs = kwargs
        self._build()

    def _build(self):
        raise NotImplementedError("build network")

    def forward(self, x):
        latent = self._torch_forward(x)
        self._set_latent(latent)
        return latent

    def _torch_forward(self, x) -> torch.tensor:
        raise NotImplementedError("define network forward")

    @property
    def latent_labels(self) -> (str, tuple, list, set, None):
        return self._latent_labels

    @latent_labels.setter
    def latent_labels(self, value: (str, tuple, list, set, None)):
        self._latent_labels = value

    @staticmethod
    def _get_activation_function(activation):
        try:
            return getattr(torch, activation, getattr(F, activation, None))
        except TypeError:
            return None

    @staticmethod
    def _activate(activation, x):
        if activation is None:
            return x
        return activation(x)

    def is_multi_latent(self):
        return self.latent_labels is not None and not isinstance(self.latent_labels, str)

    def _set_latent(self, value: torch.tensor):
        if self.latent_torch is None or self.latent is None:
            if self.latent_labels is None:
                self._latent_torch = value

            elif not self.is_multi_latent():
                self._latent_torch = {self.latent_labels: value}

            else:
                self._latent_torch = {k: v for k, v in zip(self.latent_labels, value)}

        else:
            # only overwrite existing self.latent_torch, if connection to self.latent has been established

            if self.latent_labels is None:
                self._latent_torch[...] = value

            elif not self.is_multi_latent():
                self._latent_torch[self.latent_labels][...] = value

            else:
                for label, v in zip(self.latent_labels, value):
                    self._latent_torch[label][...] = value

    @property
    def latent_torch(self) -> (None, torch.tensor, {torch.tensor}):
        """ returns encoding torch.tensor: x -> z """

        return self._latent_torch

    @property
    def latent(self) -> (ndarray, {}, {ndarray}):
        """ returns encoding numpy array: x -> z """

        try:
            if self._latent is None:
                if not isinstance(self.latent_torch, dict):
                    self._latent = self.latent_torch.to_numpy()

                else:
                    self._latent = {k: v.detach().numpy()
                                    for k, v in self.latent_torch.items()}

            return self._latent

        except AttributeError:
            return {}


class ConvEncoder(Encoder):

    def __init__(self,
                 input_dim,
                 filters: (list, tuple),
                 kernel_size: (list, tuple),
                 strides: (list, tuple),
                 latent_dim: (list, tuple, int),
                 activation: (list, tuple, str) = 'relu',
                 latent_activation: (list, tuple, str) = 'sigmoid',
                 **kwargs):

        self.input_channels_ = input_dim[0]
        self.input_dim_ = input_dim[1:]

        self.filters_ = filters
        self.kernel_size_ = kernel_size
        self.strides_ = strides
        self.activation_ = activation if not isinstance(activation, str) else [activation] * len(filters)

        self.latent_dim_ = latent_dim
        self.latent_activation_ = latent_activation if not isinstance(latent_activation, str) else [latent_activation] * len(latent_dim)

        self.conv_stack = None
        self.conv_stack_shape_out = None
        self.conv_stack_shape_in = None

        self.latent_stack = None

        super(ConvEncoder, self).__init__(**kwargs)

    def _build(self):

        self.conv_stack = []
        conv_nd = self._get_conv_nd()

        in_channels = self.input_channels_
        hw = self.input_dim_
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        for i in range(len(self.filters_)):
            f, k, s, a = self.filters_[i], self.kernel_size_[i], self.strides_[i], self.activation_[i]

            label = f'conv_{i}'
            layer = conv_nd(in_channels=in_channels, out_channels=f, kernel_size=k, stride=s)
            activation = self._get_activation_function(a)

            hw = self.conv_output_shape(hw, kernel_size=k, stride=s)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            self.conv_stack.append((label, layer, activation, out_shape))
            setattr(self, label, layer)

            in_channels = f

        self.conv_stack_shape_in = in_shape
        self.conv_stack_shape_out = out_shape

        # todo: cover None case
        self.latent_flatten = nn.Flatten()

        self.latent_stack = []
        for label, z_dim, a in zip(self.latent_labels, self.latent_dim_, self.latent_activation_):
            label = f'latent_{label}'
            layer = torch.nn.Linear(product(self.conv_stack_shape_out), z_dim)
            activation = self._get_activation_function(a)

            self.latent_stack.append((label, layer, activation, z_dim))
            setattr(self, label, layer)

    def _get_conv_nd(self):
        if len(self.input_dim_) == 1:
            conv_nd = nn.Conv1d

        elif len(self.input_dim_) == 2:
            conv_nd = nn.Conv2d

        elif len(self.input_dim_) == 3:
            conv_nd = nn.Conv3d

        else:
            raise AssertionError("input_dim must be in (1, 2, 3)")

        return conv_nd

    def _torch_forward(self, x) -> torch.tensor:
        for label, layer, activation, out_shape in self.conv_stack:
            y = layer(x)
            x = activation(y)

        x = self.latent_flatten(x)

        z = tuple([activation(layer(x)) for label, layer, activation, out_shape in self.latent_stack])

        return z[0] if self.latent_labels is None else z

    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        """
        Utility function for computing output of convolutions
        takes a number h_w or a tuple of (h,w) and returns a number h_w or a tuple of (h,w)

        see https://discuss.pytorch.org/t/utility-function-for-calculating-the-shape-of-a-conv-output/11173/6
        """

        if ndim(h_w) > 0 and not isinstance(kernel_size, tuple):
            kernel_size = tuple([kernel_size] * len(h_w))

        if ndim(h_w) > 0 and not isinstance(stride, tuple):
            stride = tuple([stride] * len(h_w))

        if ndim(h_w) > 0 and not isinstance(pad, tuple):
            pad = tuple([pad] * len(h_w))

        if not hasattr(h_w, '__len__'):
            h_w_prime = (h_w + (2 * pad) - (dilation * (kernel_size - 1)) - 1) // stride + 1

        else:
            h_w_prime = ([
                (h_w[i] + (2 * pad[i]) - (dilation * (kernel_size[i] - 1)) - 1) // stride[i] + 1
                for i in range(len(h_w))
            ])

        return h_w_prime


if __name__ == '__main__':
    input_dim = (1, 28, 28)
    z_dim = (2, )

    cnn_encoder = ConvEncoder(
        input_dim=input_dim,
        filters=(32, 64, 64, 64),
        kernel_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        activation='leaky_relu',
        latent_dim=z_dim,
        latent_labels=('z', 'mu', 'a'),
        latent_activation='sigmoid',
    )

    print(cnn_encoder)
    print('input shape     :', cnn_encoder.conv_stack_shape_in)
    print('final conv shape:', cnn_encoder.conv_stack_shape_out)
    print('latent shape    :', z_dim)

    x_random = torch.randn(1, *input_dim)
    y = cnn_encoder(x_random)

    print('output shape    :', cnn_encoder.latent_torch)
