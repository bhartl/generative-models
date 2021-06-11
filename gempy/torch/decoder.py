from numpy import ndarray, product
import torch.nn as nn
import torch.cuda
import torch.tensor
from gempy.torch.util import get_conv_transpose_nd
from gempy.torch.util import get_batch_norm_nd
from gempy.torch.util import conv_transpose_output_shape
from gempy.torch.util import call_activation
from gempy.torch.util import get_activation_function
from functools import partial


class Decoder(nn.Module):
    """ pytorch based encoder """

    def __init__(self,
                 latent_shape: (int, tuple, list),
                 latent_upscale: (int, tuple, list),
                 latent_merge: bool = True,
                 latent_activation: (str, None) = None,
                 **kwargs):
        super(Decoder, self).__init__()

        # setup latent dimensions
        self._latent_shape = None
        self.latent_shape = latent_shape

        self.latent_upscale = latent_upscale
        self.latent_merge = latent_merge  # whether to merge the latent or the final tensors

        self._latent_activation = None
        self.latent_activation = latent_activation

        self.upscale_channels = latent_upscale[0]
        self.upscale_dim = latent_upscale[1:]
        self.upscale_shape = (-1, self.upscale_channels, *self.upscale_dim)

        self.latent_stack = None

        self.kwargs = kwargs
        self._build()

    @property
    def is_multi_latent(self):
        """ Boolean describing whether multiple latent space dimensions are present """
        return hasattr(self._latent_shape, '__iter__')

    def _build(self):
        latent_shape = self._latent_shape
        latent_activation = self.latent_activation
        if not self.is_multi_latent:
            latent_shape = [latent_shape]
            latent_activation = [latent_activation] * len(latent_shape)

        self.latent_stack = []
        for i, (latent_shape, activation) in enumerate(zip(latent_shape, latent_activation)):
            label = f'decode_latent_{i}'
            layer = torch.nn.Linear(latent_shape, product(self.latent_upscale))
            activation = get_activation_function(activation)

            self.latent_stack.append((label, layer, activation, self.latent_upscale))
            setattr(self, label, layer)

    def forward(self, *x):
        if not self.is_multi_latent:
            x = x[0]

        x = [call_activation(x=layer(x[i]), foo=activation).view(self.upscale_shape)
             for i, (label, layer, activation, dim) in enumerate(self.latent_stack)]

        return x


class ConvDecoder(Decoder):
    def __init__(self,
                 filters: (tuple, list),
                 kernels_size: (tuple, list),
                 strides: (tuple, list),
                 activation: (list, tuple, str) = None,
                 padding: (int, tuple) = 1,
                 padding_mode: str = 'zeros',
                 use_batch_norm: (bool, dict) = False,
                 use_dropout: (bool, float) = False,
                 **kwargs):
        """
        :param use_batch_norm: Boolean or dict controlling whether batch-normalization
                               is applied after a convolutional layer. If parameter is
                               a dict, it is passed as kwargs to the BatchNormND layer
                               during construction.
                               Activations are performed after batch_norm layers instead
                               of directly after the convolutional filters.
        :param use_dropout: Boolean or float controlling whether dropout layers are used
                            after a Conv (and potential BatchNorm) block. A float value
                            defines the dropout rate, which defaults to 0.25.
        """

        # setup conv-kernel dimensions
        self.filters = filters
        self.kernels_size = kernels_size
        self.strides = strides
        self._activation = None
        self.activation = activation

        self.padding = padding
        self.padding_mode = padding_mode

        self.use_dropout = use_dropout
        self.use_batch_norm = use_batch_norm

        # helper variables
        self.conv_stack = None
        self.conv_stack_shape_in = None
        self.conv_stack_shape_out = None

        super(ConvDecoder, self).__init__(**kwargs)

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

        # call latent upscale decoder build
        Decoder._build(self)

        self.conv_stack = []
        conv_t = partial(get_conv_transpose_nd(self.upscale_dim), padding=self.padding, padding_mode=self.padding_mode)
        batch_norm = get_batch_norm_nd(self.upscale_dim)

        in_channels = self.upscale_channels
        hw = self.upscale_dim
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        n_conv = len(self.filters)
        for i in range(n_conv):

            # 1.) Add Convolutional layer
            f, k, s, a = self.filters[i], self.kernels_size[i], self.strides[i], self.activation[i]

            label = f'decode_conv_t_{i}'
            layer = conv_t(in_channels=in_channels, out_channels=f, kernel_size=k, stride=s)
            activation = get_activation_function(a) if (not self.use_batch_norm or i == n_conv - 1) else None

            # get shape transformation
            hw = conv_transpose_output_shape(hw, kernel_size=k, stride=s, pad=self.padding)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            # add to conv layer stack
            self.conv_stack.append((label, layer, activation, out_shape))

            # set as property (pytorch specific)
            setattr(self, label, layer)

            # 2.) Add batch normalization layer
            if self.use_batch_norm and i < n_conv - 1:
                batch_norm_label = label.replace('conv_', 'batch_norm_')
                batch_norm_kwargs = self.use_batch_norm if isinstance(self.use_batch_norm, dict) else {}
                batch_norm_layer = batch_norm(num_features=f, **batch_norm_kwargs)
                batch_norm_activation = get_activation_function(a)
                self.conv_stack.append((batch_norm_label, batch_norm_layer, batch_norm_activation, out_shape))
                setattr(self, batch_norm_label, batch_norm_layer)

            if self.use_dropout and i < n_conv - 1:
                dropout_label = label.replace('conv_', 'drop_')
                rate = 0.25 if not isinstance(self.use_dropout, float) else self.use_dropout
                dropout_layer = torch.nn.Dropout(rate)
                self.conv_stack.append((dropout_label, dropout_layer, None, out_shape))
                setattr(self, dropout_label, dropout_layer)

            in_channels = f

        self.conv_stack_shape_in = in_shape
        self.conv_stack_shape_out = out_shape

    def forward(self, *x):

        x = Decoder.forward(self, *x)

        if self.latent_merge:
            # merge all latent layers via summation, wrap in single-element list
            x = [torch.stack(x, dim=0).sum(dim=0)]

        for label, layer, activation, out_shape in self.conv_stack:
            # perform upscale stack on all inputs
            for i in range(len(x)):
                y = layer(x[i])
                x[i] = call_activation(x=y, foo=activation)

        if not self.latent_merge:
            x = torch.stack(x, dim=0).sum(dim=0)
            return x

        return x[0]


if __name__ == '__main__':

    z_shape = 2  # (2, 3)

    cnn_decoder = ConvDecoder(
        latent_shape=z_shape,
        latent_upscale=(64, 7, 7),
        filters=[64, 64, 32, 1],
        kernels_size=[3, 4, 4, 3],
        strides=[1, 2, 2, 1],
        activation=['leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid'],
        latent_merge=True,
        latent_activation=None,
        padding=1
    )

    print(cnn_decoder)

    print('latent shape    :', cnn_decoder.latent_shape)
    print('input  shape    :', cnn_decoder.conv_stack_shape_in)
    print('final conv shape:', cnn_decoder.conv_stack_shape_out)

    x_random = tuple(torch.randn(1, zi) for zi in (z_shape if hasattr(z_shape, '__iter__') else [z_shape]))
    y = cnn_decoder(*x_random)

    print('output shape    :', y.shape)
