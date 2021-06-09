from numpy import ndarray, ndim, product
import torch.nn as nn
import torch.cuda
import torch.tensor
import torch.nn.functional as F
from gempy.torch.util import conv_output_shape
from gempy.torch.util import activate

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
        if self._latent_torch is None or self._latent is None:
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
                 input_shape,
                 filters: (list, tuple),
                 kernels_size: (list, tuple),
                 strides: (list, tuple),
                 latent_dim: (list, tuple, int),
                 activation: (list, tuple, str) = 'relu',
                 latent_activation: (list, tuple, str) = 'sigmoid',
                 padding: (int, tuple) = 1,
                 padding_mode: str = 'zeros',
                 use_dropout: (bool, float) = False,
                 **kwargs):

        self.input_channels = input_shape[0]
        self.input_shape = input_shape[1:]

        self.filters = filters
        self.kernels_size = kernels_size
        self.strides = strides

        self._activation = None
        self.activation = activation

        self.padding = padding
        self.padding_mode = padding_mode

        self.use_dropout = use_dropout

        # setup latent dimensions
        self._latent_shape = None
        self.latent_shape = latent_dim

        self._latent_activation = None
        self.latent_activation = latent_activation

        self.conv_stack = None
        self.conv_stack_shape_out = None
        self.conv_stack_shape_in = None

        self.latent_stack = None

        super(ConvEncoder, self).__init__(**kwargs)

    @property
    def latent_shape(self) -> (int, tuple, list):
        return self._latent_shape

    @latent_shape.setter
    def latent_shape(self, value: (int, tuple, list)):
        if isinstance(value, int):
            value = (value,)

        self._latent_shape = value

    @property
    def latent_activation(self) -> [str]:
        return self._latent_activation

    @latent_activation.setter
    def latent_activation(self, value: (str, tuple, list)):
        if value is None or isinstance(value, str):
            value = [value] * len(self.latent_shape)

        self._latent_activation = list(value)

    @property
    def activation(self) -> [str]:
        return self._activation

    @activation.setter
    def activation(self, value: (str, tuple, list)):
        if value is None or isinstance(value, str):
            value = [value] * len(self.filters)

        self._activation = list(value)

    def _build(self):

        self.conv_stack = []
        conv_nd = self._get_conv_nd()

        in_channels = self.input_channels
        hw = self.input_shape
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        for i in range(len(self.filters)):
            f, k, s, a = self.filters[i], self.kernels_size[i], self.strides[i], self.activation[i]

            label = f'conv_{i}'
            layer = conv_nd(in_channels=in_channels,
                            out_channels=f,
                            kernel_size=k,
                            stride=s,
                            padding=self.padding,
                            padding_mode=self.padding_mode,
                            )
            activation = self._get_activation_function(a)

            hw = conv_output_shape(hw, kernel_size=k, stride=s, pad=self.padding)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            self.conv_stack.append((label, layer, activation, out_shape))
            setattr(self, label, layer)

            if self.use_dropout:
                label = label.replace('conv_', 'drop_')
                rate = 0.25 if not isinstance(self.use_dropout, float) else self.use_dropout
                layer = torch.nn.Dropout(rate)
                self.conv_stack.append((label, layer, None, out_shape))
                setattr(self, label, layer)

            in_channels = f

        self.conv_stack_shape_in = in_shape
        self.conv_stack_shape_out = out_shape

        # todo: cover None case
        self.latent_flatten = nn.Flatten()

        self.latent_stack = []
        for i, latent_shape in enumerate(self.latent_shape):
            try:
                label = self.latent_labels[i]
            except (TypeError, KeyError):
                label = i

            label = f'latent_{label}'
            layer = torch.nn.Linear(product(self.conv_stack_shape_out), latent_shape)
            activation = self._get_activation_function(self.latent_activation[i])

            self.latent_stack.append((label, layer, activation, latent_shape))
            setattr(self, label, layer)

    def _get_conv_nd(self):
        if len(self.input_shape) == 1:
            conv_nd = nn.Conv1d

        elif len(self.input_shape) == 2:
            conv_nd = nn.Conv2d

        elif len(self.input_shape) == 3:
            conv_nd = nn.Conv3d

        else:
            raise AssertionError("input_dim must be in (1, 2, 3)")

        return conv_nd

    def _torch_forward(self, x) -> torch.tensor:
        for label, layer, activation, out_shape in self.conv_stack:
            y = layer(x)
            x = activate(x=y, foo=activation)

        x = self.latent_flatten(x)
        z = tuple([activate(x=layer(x), foo=activation)
                   for label, layer, activation, out_shape in self.latent_stack])

        if self.latent_labels is None and len(z) == 1:
            return z[0]

        return z


if __name__ == '__main__':
    input_shape = (1, 28, 28)
    z_shape = 2  # (2, 3)

    cnn_encoder = ConvEncoder(
        input_shape=input_shape,
        filters=(32, 64, 64, 64),
        kernels_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        activation='leaky_relu',
        latent_dim=z_shape,
        # latent_labels=('z', 'mu'),
        latent_labels=None,
        latent_activation='sigmoid',
        use_dropout=True,
    )

    print(cnn_encoder)
    print('input shape     :', cnn_encoder.conv_stack_shape_in)
    print('final conv shape:', cnn_encoder.conv_stack_shape_out)
    print('latent shape    :', cnn_encoder.latent_shape)

    x_random = torch.randn(1, *input_shape)
    y = cnn_encoder(x_random)

    print('output shape    :', cnn_encoder.latent_torch)
