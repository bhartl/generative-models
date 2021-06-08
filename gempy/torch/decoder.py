from numpy import ndarray, ndim, product
import torch.nn as nn
import torch.cuda
import torch.tensor
import torch.nn.functional as F


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
                 latent_dim: (int, tuple, list),
                 latent_upscale: (int, tuple, list),
                 filters: (tuple, list),
                 kernel_size: (tuple, list),
                 strides: (tuple, list),
                 activation: (list, tuple, str) = 'leaky_relu',
                 latent_merge: bool = True,
                 latent_activation: (str, None) = None,
                 **kwargs):

        self.latent_dim_ = latent_dim
        self.latent_upscale_ = latent_upscale
        self.latent_merge_ = latent_merge  # whether to merge the latent or the final tensors
        self.latent_activation_ = latent_activation if latent_activation is not None and not isinstance(latent_activation, str) else [latent_activation] * len(latent_dim)

        self.upscale_dim_ = latent_upscale[1:]
        self.upscale_channels_ = latent_upscale[0]

        self.filters_ = filters
        self.kernel_size_ = kernel_size
        self.strides_ = strides
        self.activation_ = activation if not isinstance(activation, str) else [activation] * len(filters)

        self.conv_stack = None
        self.conv_stack_shape_out = None
        self.conv_stack_shape_in = None

        self.conv_stack = None

        super(ConvDecoder, self).__init__(**kwargs)

    def _build(self):
        latent_dim = self.latent_dim_
        latent_activation = self.latent_activation_
        if not self.is_multi_latent():
            latent_dim = [latent_dim]
            latent_activation = [latent_activation] * len(latent_dim)

        self.latent_stack = []
        for i, (z_dim, activation) in enumerate(zip(latent_dim, latent_activation)):
            label = f'decode_latent_{i}'
            layer = torch.nn.Linear(z_dim, product(self.latent_upscale_))
            activation = self._get_activation_function(activation)

            self.latent_stack.append((label, layer, activation, self.latent_upscale_))
            setattr(self, label, layer)

        self.upscale_shape = (-1, self.upscale_channels_, *self.upscale_dim_)

        self.conv_stack = []
        conv_transpose = self._get_conv_transpose_nd()

        in_channels = self.upscale_channels_
        hw = self.upscale_dim_
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        for i in range(len(self.filters_)):
            f, k, s, a = self.filters_[i], self.kernel_size_[i], self.strides_[i], self.activation_[i]

            label = f'decode_conv_t_{i}'
            pad = 0
            layer = conv_transpose(in_channels=in_channels, out_channels=f, kernel_size=k, stride=s, padding=pad)
            activation = self._get_activation_function(a)

            hw = self.conv_transpose_output_shape(hw, kernel_size=k, stride=s, pad=pad)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            self.conv_stack.append((label, layer, activation, out_shape))
            setattr(self, label, layer)

            in_channels = f

        self.conv_stack_shape_in = in_shape
        self.conv_stack_shape_out = out_shape

    def is_multi_latent(self):
        return hasattr(self.latent_dim_, '__iter__')

    def _get_conv_transpose_nd(self):
        if len(self.upscale_dim_) == 1:
            conv_transpose_nd = nn.ConvTranspose1d

        elif len(self.upscale_dim_) == 2:
            conv_transpose_nd = nn.ConvTranspose2d

        elif len(self.upscale_dim_) == 3:
            conv_transpose_nd = nn.ConvTranspose3d

        else:
            raise AssertionError("input_dim must be in (1, 2, 3)")

        return conv_transpose_nd

    @staticmethod
    def conv_transpose_output_shape(h_w, kernel_size=1, stride=1, pad=0):
        """
        Utility function for computing output of transposed convolutions
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
            h_w_prime = (h_w - 1) * stride - 2 * pad + kernel_size + pad

        else:
            h_w_prime = ([
                (h_w[i] - 1) * stride[i] - 2 * pad[i] + kernel_size[i] + pad[i]
                for i in range(len(h_w))
            ])

        return h_w_prime

    def forward(self, *x):
        if not self.is_multi_latent():
            x = x[0]

        x = [self._activate(activation=activation, x=layer(xi)).view(self.upscale_shape)
             for xi, (label, layer, activation, dim) in zip(x, self.latent_stack)]

        if self.latent_merge_:
            # merge all latent layers via summation, wrap in single-element list
            x = [torch.stack(x, dim=0).sum(dim=0)]

        for label, layer, activation, out_shape in self.conv_stack:
            # perform upscale stack on all inputs
            for i in range(len(x)):
                y = layer(x[i])
                x[i] = self._activate(activation=activation, x=y)
                # print(x[i].shape, y.shape)  # TODO: correct shaping

        x = torch.stack(x, dim=0).sum(dim=0) if not self.latent_merge_ else x[0]

        return x


if __name__ == '__main__':

    z_dim = (2, )

    cnn_decoder = ConvDecoder(
        latent_dim=z_dim,
        latent_upscale=(64, 3, 3),
        filters=[64, 64, 32, 1],
        kernel_size=[3, 4, 4, 3],
        strides=[1, 2, 2, 1],
        activation='leaky_relu',
        latent_merge=True,
        latent_activation=None,
    )

    print(cnn_decoder)

    print('latent shape    :', z_dim)
    print('input  shape:', cnn_decoder.conv_stack_shape_in)
    print('final conv shape:', cnn_decoder.conv_stack_shape_out)

    x_random = torch.randn(1, *z_dim)
    y = cnn_decoder(x_random)

    print('output shape    :', y.shape)
