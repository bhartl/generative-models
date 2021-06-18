class Decoder(object):
    """ abstract decoder: z (latent space) -> x_hat (reconstructed feature space)

    Usage and Inheritance:

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
        super(Decoder, self).__init__()

        # setup latent dimensions
        self._latent_dim = None
        self.latent_dim = latent_dim

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
    def latent_dim(self) -> (int, tuple, list):
        """ number of dimensions per latent space output """
        return self._latent_dim

    @latent_dim.setter
    def latent_dim(self, value: (int, tuple, list)):
        """ number of dimensions per latent space output """
        if isinstance(value, int):
            value = (value,)

        self._latent_dim = value

    @property
    def latent_activation(self) -> [str]:
        """ activation function(s) of the latent space, can be `None` """
        return self._latent_activation

    @latent_activation.setter
    def latent_activation(self, value: (str, tuple, list)):
        """ activation function(s) of the latent space, can be `None` """
        if value is None or isinstance(value, str):
            value = [value] * len(self.latent_dim)

        self._latent_activation = list(value)

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

        raise NotImplementedError('build latent')

    @property
    def is_multi_latent(self) -> bool:
        """ Boolean describing whether multiple latent space dimensions are present """
        return hasattr(self._latent_dim, '__iter__')

    def forward(self, *x):
        """ pytorch forward-method performing the encoding.

        :param x: input (latent space) tensor
        :returns: reconstructed feature space tensor if the `latent_merge` property is set,
                  or a tuple of `upscaled` feature space tensors, one for each latent space dimension.
        """
        raise NotImplementedError('forward')

