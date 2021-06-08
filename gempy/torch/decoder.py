from numpy import ndarray, product
import torch.nn as nn
import torch.cuda
import torch.tensor
import torch.nn.functional as F
from gempy.torch.util import conv_transpose_output_shape


class Decoder(nn.Module):
    """ pytorch based encoder """

    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

        self.kwargs = kwargs
        self._build()

    def _build(self):
        raise NotImplementedError("build network")

    def forward(self, x):
        raise NotImplementedError("define network forward")

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


class ConvDecoder(Decoder):
    def __init__(self,
                 latent_shape: (int, tuple, list),
                 latent_upscale: (int, tuple, list),
                 filters: (tuple, list),
                 kernels_size: (tuple, list),
                 strides: (tuple, list),
                 activation: (list, tuple, str) = None,
                 latent_merge: bool = True,
                 latent_activation: (str, None) = None,
                 padding: (int, tuple) = 1,
                 padding_mode: str = 'zeros',
                 **kwargs):

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

        # setup conv-kernel dimensions
        self.filters = filters
        self.kernels_size = kernels_size
        self.strides = strides
        self._activation = None
        self.activation = activation

        self.padding = padding
        self.padding_mode = padding_mode

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
        latent_shape = self._latent_shape
        latent_activation = self.latent_activation
        if not self.is_multi_latent():
            latent_shape = [latent_shape]
            latent_activation = [latent_activation] * len(latent_shape)

        self.latent_stack = []
        for i, (latent_shape, activation) in enumerate(zip(latent_shape, latent_activation)):
            label = f'decode_latent_{i}'
            layer = torch.nn.Linear(latent_shape, product(self.latent_upscale))
            activation = self._get_activation_function(activation)

            self.latent_stack.append((label, layer, activation, self.latent_upscale))
            setattr(self, label, layer)

        self.conv_stack = []
        conv_transpose = self._get_conv_transpose_nd()

        in_channels = self.upscale_channels
        hw = self.upscale_dim
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        for i in range(len(self.filters)):
            f, k, s, a = self.filters[i], self.kernels_size[i], self.strides[i], self.activation[i]

            label = f'decode_conv_t_{i}'
            layer = conv_transpose(in_channels=in_channels,
                                   out_channels=f,
                                   kernel_size=k,
                                   stride=s,
                                   padding=self.padding,
                                   padding_mode=self.padding_mode,
                                   )
            activation = self._get_activation_function(a)

            hw = conv_transpose_output_shape(hw, kernel_size=k, stride=s, pad=self.padding)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            self.conv_stack.append((label, layer, activation, out_shape))
            setattr(self, label, layer)

            in_channels = f

        self.conv_stack_shape_in = in_shape
        self.conv_stack_shape_out = out_shape

    def is_multi_latent(self):
        return hasattr(self._latent_shape, '__iter__')

    def _get_conv_transpose_nd(self):
        if len(self.upscale_dim) == 1:
            conv_transpose_nd = nn.ConvTranspose1d

        elif len(self.upscale_dim) == 2:
            conv_transpose_nd = nn.ConvTranspose2d

        elif len(self.upscale_dim) == 3:
            conv_transpose_nd = nn.ConvTranspose3d

        else:
            raise AssertionError("input_shape must be in (1, 2, 3)")

        return conv_transpose_nd

    def forward(self, *x):
        if not self.is_multi_latent():
            x = x[0]

        x = [self._activate(activation=activation, x=layer(x[i])).view(self.upscale_shape)
             for i, (label, layer, activation, dim) in enumerate(self.latent_stack)]

        if self.latent_merge:
            # merge all latent layers via summation, wrap in single-element list
            x = [torch.stack(x, dim=0).sum(dim=0)]

        for label, layer, activation, out_shape in self.conv_stack:
            # perform upscale stack on all inputs
            for i in range(len(x)):
                y = layer(x[i])
                x[i] = self._activate(activation=activation, x=y)

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
