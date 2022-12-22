from numpy import product
import torch.nn as nn
import torch.cuda
import torch.tensor
from gempy.decoder import Decoder as ABCDecoder
from gempy.torch_.util import get_conv_transpose_nd
from gempy.torch_.util import get_batch_norm_nd
from gempy.torch_.util import conv_transpose_output_shape
from gempy.torch_.util import call_activation
from gempy.torch_.util import get_activation_function
from functools import partial


class Decoder(ABCDecoder, nn.Module):
    """ pytorch based decoder: z (latent space) -> x_hat (reconstructed feature space)

    Usage and Inheritance:

    - Inherits from `torch.nn.Module`
    - `_build` routine is to be overwritten by more specific `Decoder` child-classes to define the network
    - `_torch_forward` routine is to be overwritten by more specific `Decoder` child-classes to define the tensor flow

    Comments:

    - multiple latent space layers/dimensions are possible
    """

    def __init__(self,
                 latent_dim: (int, tuple, list),
                 latent_upscale: (tuple, list),
                 latent_activation: (str, None) = None,
                 latent_merge: bool = True,
                 **kwargs):
        """ Constructs a `Decoder` instance

        :param latent_dim: Integer or tuple of integers, defining the size of each latent space output.
        :param latent_upscale: Iterable defining the shape of the output dense layer applied to the latent space input.
        :param latent_activation: String or boolean defining the activation of the latent space dense layer
                                  (defaults to None).
        :param latent_merge: Boolean controlling whether latent space outputs should be merged (per default)
                             or whether each latent space output should be upscaled independently.
        :param kwargs: Additional keyword arguments.
        """
        nn.Module.__init__(self)
        ABCDecoder.__init__(self,
                            latent_dim=latent_dim,
                            latent_upscale=latent_upscale,
                            latent_activation=latent_activation,
                            latent_merge=latent_merge,
                            **kwargs)

    def _build(self):
        """ helper function to build the network:

        A stack of dense (torch.nn.Linear) latent space layers are defined,
        each with their own activation function (which defaults to None) which
        transform the (different) latent space input into the `upscaled` output
        of the `Decoder`.

        The layers are collected in the `Decoder`-instance's `latent_stack` property.

        The layers are added as properties and incrementally labeled via
        'decode_latent_{label_i}' if labels are provided or 'decode_latent_i' otherwise.
        """

        latent_dim = self._latent_dim
        latent_activation = self.latent_activation
        if not self.is_multi_latent:
            latent_dim = [latent_dim]
            latent_activation = [latent_activation] * len(latent_dim)

        self.latent_stack = []
        for i, (latent_dim, activation) in enumerate(zip(latent_dim, latent_activation)):
            label = f'decode_latent_{i}'
            layer = torch.nn.Linear(latent_dim, product(self.latent_upscale))
            activation = get_activation_function(activation)

            self.latent_stack.append((label, layer, activation, self.latent_upscale))
            setattr(self, label, layer)

    def forward(self, *x: (tuple, torch.tensor)) -> (list, torch.tensor):
        """ pytorch forward-method performing the encoding.

        :param x: input (latent space) tensor
        :returns: reconstructed feature space tensor if the `latent_merge` property is set,
                  or a tuple of `upscaled` feature space tensors, one for each latent space dimension.
        """
        if not self.is_multi_latent:
            x = x[0]

        x = tuple(call_activation(x=layer(x[i]), foo=activation).view(self.upscale_shape)
                  for i, (label, layer, activation, dim) in enumerate(self.latent_stack))

        # x_decode = []
        # for i, (label, layer, activation, dim) in enumerate(self.latent_stack):
        #     print(x.shape, i)
        #     xi = call_activation(x=layer(x[i]), foo=activation).view(self.upscale_shape)
        #     x_decode.append(xi)
        # x = x_decode

        if self.latent_merge:
            x = torch.stack(x, dim=0).sum(dim=0)

        return x


class ConvTDecoder(Decoder):
    """ pytorch based convolutional-transpose decoder:
        z (latent space) -> [UPSCALE] -> [CNN_T] -> x_hat (feature space)

    Usage and Inheritance:

    - Inherits from `Decoder`
    - `_build` routine is to be overwritten by more specific `Decoder` child-classes to define the network
    - `_torch_forward` routine is to be overwritten by more specific `Decoder` child-classes to define the tensor flow

    Comments:

    - multiple latent space layers/dimensions are possible
    - a stack of (filter, kernel, stride, activation) can be provided, to quickly build a CNN Encoder
    - 1D, 2D and 3D Convolutions are implemented (one of a kind vor a given ConvEncoder)
    - the latent space layers can be merged after UPSCALING or after the entire CNN network using the `latent_merge`
      flag
    """

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
        """ Constructs a ConvEncoder instance

        :param filters: List or tuple of integers, specifying the number of filters per conv layer
        :param kernels_size: List or tuple of kernel sizes for each conv layer
        :param strides: List or tuple of stride values for each conv layer
        :param activation: String-Name or list of names of activation functions to be used in each conv layer
                           (defaults to 'relu').
                           If a single name is specified, the activation is used for each layer.
                           If None is provided, no activation is used.
        :param padding: Padding of all conv layers (defaults to 1, global property)
        :param padding_mode: Padding mode (defaults to 'zeros', see torch doc)        :param use_batch_norm: Boolean or dict controlling whether batch-normalization
                               is applied after a convolutional layer. If parameter is
                               a dict, it is passed as kwargs to the BatchNormND layer
                               during construction.
                               Activations are performed after batch_norm layers instead
                               of directly after the convolutional filters.
        :param use_dropout: Boolean or float controlling whether dropout layers are used
                            after a Conv (and potential BatchNorm) block. A float value
                            defines the dropout rate, which defaults to 0.25.
        :param kwargs: Keyword Arguments forwarded to the super constructor (see Decoder)
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
        self.deconv_stack = None
        self.deconv_stack_shape_in = None
        self.deconv_stack_shape_out = None

        super(ConvTDecoder, self).__init__(**kwargs)

    @property
    def activation(self) -> [str]:
        """ activation function(s) of the convolutional layers """
        return self._activation

    @activation.setter
    def activation(self, value: (str, tuple, list)):
        """ activation function(s) of the convolutional layers """
        if value is None or isinstance(value, str):
            value = [value] * len(self.filters)

        self._activation = list(value)

    def _build(self):
        """ helper function to build the network:

        A stack of groups of (convolutional-transpose, batch normalization, dropout) layers are generated,
        each stack has its own activation function. Both batch normalization and dropout are only
        used, if the properties `use_batch_norm` and `use_dropout` are set.
        `use_batch_norm` may be interpreted as kwargs forwarded to the BatchNormND torch layers,
        `use_dropout` may be interpreted as float specifying the dropout rate.

        The conv-transpose/batch norm/dropout layers (and metadata) are collected in the `ConvTDencoder`-instance's
        `conv_transpose_stack` property.

        The initial and final nd-shape of the tensors are provided in the

        - `conv_transpose_stack_shape_in` and
        - `conv_transpose_stack_shape_out`

        properties.

        First, the latent space is upscaled via denslely connected layers into the shape given by `latent_upscale`.
        The the stack of conv_transpose/batch norm/dropout layers is applied.

        If latent_merge is set, all latent outputs are merged together via addition directly after upscaling,
        otherwise the final conv-transpose outputs are merged.

        All layers are added as properties and incrementally labeled via

        - 'decode_latent_{label_i}' for latent layers
        - `decode_conv_t_{label_i} for convolutional layers
        - `decode_batch_norm_{label_i} for batch normalization layers
        - 'decode_dropout_{label_i} for batch dropout layers
        """

        # call latent upscale decoder build
        Decoder._build(self)

        self.deconv_stack = []
        conv_t = partial(get_conv_transpose_nd(self.upscale_dim), padding=self.padding, padding_mode=self.padding_mode)
        batch_norm = get_batch_norm_nd(self.upscale_dim)

        in_channels = self.upscale_channels
        hw = self.upscale_dim
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        # Add a stack of single ConvTranspose-Batch Normalize-Dropout layers
        n_conv = len(self.filters)
        for i in range(n_conv):

            # 1.) Add ConvTranspose layer
            f, k, s, a = self.filters[i], self.kernels_size[i], self.strides[i], self.activation[i]

            label = f'deconv_{i}'
            layer = conv_t(in_channels=in_channels, out_channels=f, kernel_size=k, stride=s)
            activation = get_activation_function(a) if (not self.use_batch_norm or i == n_conv - 1) else None

            # get shape transformation
            hw = conv_transpose_output_shape(hw, kernel_size=k, stride=s, pad=self.padding)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            # add to conv-transpose layer stack
            self.deconv_stack.append((label, layer, activation, out_shape))

            # set as property (pytorch specific)
            setattr(self, label, layer)

            # 2.) Add batch normalization layer
            if self.use_batch_norm and i < n_conv - 1:
                batch_norm_label = label.replace('deconv_', 'debatch_norm_')
                batch_norm_kwargs = self.use_batch_norm if isinstance(self.use_batch_norm, dict) else {}
                batch_norm_layer = batch_norm(num_features=f, **batch_norm_kwargs)
                batch_norm_activation = get_activation_function(a)
                self.deconv_stack.append((batch_norm_label, batch_norm_layer, batch_norm_activation, out_shape))
                setattr(self, batch_norm_label, batch_norm_layer)

            # 3.) Add dropout layer
            if self.use_dropout and i < n_conv - 1:
                dropout_label = label.replace('deconv_', 'dedrop_')
                rate = 0.25 if not isinstance(self.use_dropout, float) else self.use_dropout
                dropout_layer = torch.nn.Dropout(rate)
                self.deconv_stack.append((dropout_label, dropout_layer, None, out_shape))
                setattr(self, dropout_label, dropout_layer)

            # the out channels from the current stack become the in channels from the next stack
            in_channels = f

        # remember the input and final output shape of the conv-transpose stacks
        self.deconv_stack_shape_in = in_shape
        self.deconv_stack_shape_out = out_shape

    def forward(self, *x):
        """ torch forward method,
            reconstructs an original unknown, feature-rich input based on latent representations *x

        :param x: torch input tensor to the `Decoder`
        :returns: reconstruction x_hat of the original input
        """

        x = Decoder.forward(self, *x)

        # all latent layers merged by decoder, wrap in single-element tuple
        x = [x] if self.latent_merge else list(x)

        for label, layer, activation, out_shape in self.deconv_stack:
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

    cnn_decoder = ConvTDecoder(
        latent_dim=z_shape,
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

    print('latent shape    :', cnn_decoder.latent_dim)
    print('input  shape    :', cnn_decoder.deconv_stack_shape_in)
    print('final conv shape:', cnn_decoder.deconv_stack_shape_out)

    x_random = tuple(torch.randn(1, zi) for zi in (z_shape if hasattr(z_shape, '__iter__') else [z_shape]))
    y = cnn_decoder(*x_random)

    print('output shape    :', y.shape)
