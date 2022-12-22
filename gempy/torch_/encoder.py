from numpy import ndarray, product
import torch.nn as nn
import torch.cuda
import torch.tensor
from gempy.encoder import Encoder as ABCEncoder
from gempy.torch_.util import get_conv_nd
from gempy.torch_.util import get_batch_norm_nd
from gempy.torch_.util import conv_output_shape
from gempy.torch_.util import call_activation
from gempy.torch_.util import get_activation_function
from functools import partial


class Encoder(ABCEncoder, nn.Module):
    """ pytorch based encoder: x (feature space) -> z (latent space)

    Usage and Inheritance:

    - Inherits from `torch.nn.Module`
    - `_build` routine is to be overwritten by more specific `Encoder` child-classes to define the network
    - `_torch_forward` routine is to be overwritten by more specific `Encoder` child-classes to define the tensor flow

    Comments:

    - multiple latent space layers/dimensions are possible
    - retrieval of evaluations possible via dictionary mapping (if labels are provided)
    """

    def __init__(self,
                 latent_dim: (list, tuple, int) = 2,
                 latent_labels: (str, tuple, list, set, None) = 'z',
                 latent_activation: (list, tuple, str) = None,
                 latent_track: bool = False,
                 **kwargs):
        """ Constructs a `Encoder` instance

        :param latent_dim: Integer or tuple of integers, defining the size of each latent space output (defaults to 2).
        :param latent_labels: String or list of strings labeling the latent dimensions (defaults to 'z').
                              Labels can be used to retrieve the latent space evaluations in the `latent_torch` or
                              `latent` properties (i.e., dictionaries).
                              If label is`None`, no dict-wrapping will be applied to the `latent_torch` or `latent`
                              properties.
        :param latent_activation: String or None, defining the torch-activation which should be used after the latent
                                  space layers (defaults to `None`).
        :param latent_labels: Label or list of labels, specifying the number of outputs of the encoder (defaults to 'z').
                              This is important for the Variational Auto Encoder, where the Encoder must provide both,
                              a mean and a (log-) standard deviation value as output, i.e., two latent labels.
                              If latent_labels is `None` -> the latent space will be directly addressed as tensor/numpy
                              array, otherwise a it will be wrapped as a dictionary with keys provided by latent_labels.
        :param latent_track: Boolean controlling whether the latent space evaluations are tracked in the `latent` and
                             `latent_torch` properties.
        :param kwargs: Additional kwargs which might be used.
        """
        nn.Module.__init__(self)

        ABCEncoder.__init__(self,
                            latent_dim=latent_dim,
                            latent_labels=latent_labels,
                            latent_activation=latent_activation,
                            latent_track=latent_track,
                            **kwargs)

        self._latent_torch = None

    def _build(self):
        """ helper function to build the network:

        A stack of dense (torch.nn.Linear) latent space layers are defined,
        each with their own activation function (which defaults to None).

        The layers are collected in the `Encoder`-instance's `latent_stack` property.

        The layers are added as properties and incrementally labeled via
        'latent_{label_i}' if labels are provided or 'latent_i' otherwise.
        """
        self.latent_stack = []
        for i, latent_dim in enumerate(self.latent_dim):
            try:
                label = self.latent_labels[i]
            except (TypeError, KeyError):
                label = i

            label = f'latent_{label}'
            layer = torch.nn.Linear(product(self.conv_stack_shape_out), latent_dim)
            activation = get_activation_function(self.latent_activation[i])

            self.latent_stack.append((label, layer, activation, latent_dim))
            setattr(self, label, layer)

    def encoder_forward(self, x) -> torch.tensor:
        """ Encoder forward-method performing the encoding.

        :param x: input tensor
        :returns: latent space tensor (if no labels are defined or if only one latent space dimension is present),
                  or tuple of latent space tensors, one for each latent space dimension.
        """
        z = tuple(call_activation(x=layer(x), foo=activation)
                  for label, layer, activation, out_shape in self.latent_stack)

        if self.latent_labels is None and len(z) == 1:
            return z[0]

        return z

    def set_latent(self, value: torch.tensor):
        """ setter for latent space evaluation property

        :param value: torch-tensor whose values are stored in the `latent_torch` property.

        Updates the `latent_torch` torch-tensor property with the values in `value`.

        Updates the `latent` numpy tensor property, if it has been retrieved before.
        """

        if self._latent_torch is None or self._latent is None:
            # neither latent_torch nor latent have been retrieved yet

            if self.latent_labels is None:
                # set torch tensor directly, no dict-wrapping
                self._latent_torch = value

            elif not self.is_multi_latent:
                # single labeled latent dimensions, use dict wrapping
                self._latent_torch = {self.latent_labels: value}

            else:
                # multiple latent dimensions and labels present, use dict wrapping
                self._latent_torch = {k: v for k, v in zip(self.latent_labels, value)}

        else:
            # only overwrite existing self.latent_torch, if connection to self.latent has been established
            if self.latent_labels is None:
                self._latent_torch[...] = value

            elif not self.is_multi_latent:
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
    """ pytorch based convolutional encoder: x (feature space) -> [CNN] -> z (latent space)

    Usage and Inheritance:

    - Inherits from `Encoder`
    - `_build` routine is to be overwritten by more specific `Encoder` child-classes to define the network
    - `_torch_forward` routine is to be overwritten by more specific `Encoder` child-classes to define the tensor flow

    Comments:

    - multiple latent space layers/dimensions are possible
    - retrieval of evaluations possible via dictionary mapping (if labels are provided)
    - a stack of (filter, kernel, stride, activation) can be provided, to quickly build a CNN Encoder
    - 1D, 2D and 3D Convolutions are implemented (one of a kind vor a given ConvEncoder)
    - the last convolutional dimension will be flattened and will be forwarded to the `Encoder.encoder_forward` method.
    """

    def __init__(self,
                 input_shape,
                 filters: (list, tuple),
                 kernels_size: (list, tuple),
                 strides: (list, tuple),
                 activation: (list, tuple, str) = 'relu',
                 padding: (int, tuple) = 1,
                 padding_mode: str = 'zeros',
                 use_batch_norm: (bool, dict) = False,
                 use_dropout: (bool, float) = False,
                 **kwargs):
        """ Constructs a ConvEncoder instance

        :param input_shape: list of integers specifying the input shape dimensions (channels, x, y, ...)
                            1D, 2D and 3D convolutions are implemented.
        :param filters: List or tuple of integers, specifying the number of filters per conv layer
        :param kernels_size: List or tuple of kernel sizes for each conv layer
        :param strides: List or tuple of stride values for each conv layer
        :param activation: String-Name or list of names of activation functions to be used in each conv layer
                           (defaults to 'relu').
                           If a single name is specified, the activation is used for each layer.
                           If None is provided, no activation is used.
        :param padding: Padding of all conv layers (defaults to 1, global property)
        :param padding_mode: Padding mode (defaults to 'zeros', see torch doc)
        :param use_batch_norm: Boolean or dict controlling whether batch-normalization
                               is applied after a convolutional layer. If parameter is
                               a dict, it is passed as kwargs to the BatchNormND layer
                               during construction.
                               Activations are performed after batch_norm layers instead
                               of directly after the convolutional filters.
        :param use_dropout: Boolean or float controlling whether dropout layers are used
                            after a Conv (and potential BatchNorm) block. A float value
                            defines the dropout rate, which defaults to 0.25.
        :param kwargs: Keyword Arguments forwarded to the super constructor (see Encoder)
        """

        self.input_channels = input_shape[0]
        self.input_shape = input_shape[1:]

        self.filters = filters
        self.kernels_size = kernels_size
        self.strides = strides

        self._activation = None
        self.activation = activation

        self.padding = padding
        self.padding_mode = padding_mode

        self.use_batch_norm = use_batch_norm
        self.use_dropout = use_dropout

        self.conv_stack = None
        self.conv_stack_shape_out = None
        self.conv_stack_shape_in = None

        super(ConvEncoder, self).__init__(**kwargs)

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

        A stack of groups of (convolutional, batch normalization, dropout) layers are generated, 
        each stack has its own activation function. Both batch normalization and dropout are only
        used, if the properties `use_batch_norm` and `use_dropout` are set.
        `use_batch_norm` may be interpreted as kwargs forwarded to the BatchNormND torch layers,
        `use_dropout` may be interpreted as float specifying the dropout rate.

        The conv/batch norm/dropout layers (and metadata) are collected in the `ConvEncoder`-instance's
        `conv_stack` property.

        The initial and final nd-shape of the tensors are provided in the

        - `conv_stack_shape_in` and
        - `conv_stack_shape_out`

        properties.

        The final conv layer is flattened and the flattened tensor is forwarded to the Encoder's `latent_stack` layers.

        All layers are added as properties and incrementally labeled via

        - `conv_{label_i} for convolutional layers
        - `batch_norm_{label_i} for batch normalization layers
        - 'dropout_{label_i} for batch dropout layers
        - 'latent_{label_i}' for latent layers
        """

        self.conv_stack = []
        conv_nd = partial(get_conv_nd(self.input_shape), padding=self.padding, padding_mode=self.padding_mode)
        batch_norm = get_batch_norm_nd(self.input_shape)

        in_channels = self.input_channels
        hw = self.input_shape
        in_shape = tuple([xyz
                          for shape_lists in [in_channels, hw]
                          for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])
        out_shape = None

        # Add a stack of single Conv-Batch Normalize-Dropout layers
        for i in range(len(self.filters)):

            # 1.) Add Convolutional layer
            f, k, s, a = self.filters[i], self.kernels_size[i], self.strides[i], self.activation[i]

            label = f'conv_{i}'
            layer = conv_nd(in_channels=in_channels, out_channels=f, kernel_size=k, stride=s)
            activation = get_activation_function(a) if not self.use_batch_norm else None

            # get shape transformation
            hw = conv_output_shape(hw, kernel_size=k, stride=s, pad=self.padding)
            out_shape = tuple([xyz
                              for shape_lists in [f, hw]
                              for xyz in (shape_lists if hasattr(shape_lists, '__iter__') else [shape_lists])])

            # add to conv layer stack
            self.conv_stack.append((label, layer, activation, out_shape))

            # set as property (pytorch specific)
            setattr(self, label, layer)

            # 2.) Add batch normalization layer
            if self.use_batch_norm:
                batch_norm_label = label.replace('conv_', 'batch_norm_')
                batch_norm_kwargs = self.use_batch_norm if isinstance(self.use_batch_norm, dict) else {}
                batch_norm_layer = batch_norm(num_features=f, **batch_norm_kwargs)
                batch_norm_activation = get_activation_function(a)
                self.conv_stack.append((batch_norm_label, batch_norm_layer, batch_norm_activation, out_shape))
                setattr(self, batch_norm_label, batch_norm_layer)

            # 3.) Add dropout layer
            if self.use_dropout:
                dropout_label = label.replace('conv_', 'drop_')
                rate = 0.25 if not isinstance(self.use_dropout, float) else self.use_dropout
                dropout_layer = torch.nn.Dropout(rate)
                self.conv_stack.append((dropout_label, dropout_layer, None, out_shape))
                setattr(self, dropout_label, dropout_layer)

            # the out channels from the current stack become the in channels from the next stack
            in_channels = f

        # remember the input and final output shape of the conv stacks
        self.conv_stack_shape_in = in_shape
        self.conv_stack_shape_out = out_shape

        # flatten the final conv layer
        self.latent_flatten = nn.Flatten()

        # build the latent space dense network
        Encoder._build(self)

    def encoder_forward(self, x: torch.tensor) -> (torch.tensor, tuple):
        """ ConvEncoder forward routine to evaluate the latent space given an input tensor x

        :param x: input torch tensor
        :returns: latent space output, either as torch tensor if no labels are given or as tuple with a tensor per label
        """

        # sequentially apply conv-stack layers
        for label, layer, activation, out_shape in self.conv_stack:
            y = layer(x)
            x = call_activation(x=y, foo=activation)

        # flatten final conv-stack layer output
        x = self.latent_flatten(x)

        # apply latent space transformation
        z = Encoder.encoder_forward(self, x)

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
        latent_labels=None,  # ('z', 'mu'),
        latent_activation='sigmoid',
        use_dropout=True,
    )

    print(cnn_encoder)
    print('input shape     :', cnn_encoder.conv_stack_shape_in)
    print('final conv shape:', cnn_encoder.conv_stack_shape_out)
    print('latent shape    :', cnn_encoder.latent_dim)

    x_random = torch.randn(1, *input_shape)
    y = cnn_encoder(x_random)

    print('output shape    :', cnn_encoder.latent_torch)
