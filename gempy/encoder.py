from numpy import ndarray


class Encoder(object):
    """ abstract encoder: x (feature space) -> z (latent space)

    Usage and Inheritance:

    - `_build` routine is to be overwritten by more specific `Encoder` child-classes to define the network
    - `encoder_forward` routine is to be overwritten by more specific `Encoder` child-classes to define the tensor flow

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
                              Labels can be used to retrieve the latent space evaluations in `latent` properties (i.e.,
                              dictionaries).
                              If label is`None`, no dict-wrapping will be applied to the `latent` properties.
        :param latent_activation: String or None, defining the activation functions which should be used after the
                                  latent space layers (defaults to `None`).
        :param latent_labels: Label or list of labels, specifying the number of outputs of the encoder (defaults to 'z').
                              This is important for the Variational Auto Encoder, where the Encoder must provide both,
                              a mean and a (log-) standard deviation value as output, i.e., two latent labels.
                              If latent_labels is `None` -> the latent space will be directly addressed as tensor/numpy
                              array, otherwise a it will be wrapped as a dictionary with keys provided by latent_labels.
        :param latent_track: Boolean controlling whether the latent space evaluations are tracked in the `latent`
                             property.
        :param kwargs: Additional kwargs which might be used.
        """

        # setup latent dimensions
        self._latent_dim = None
        self.latent_dim = latent_dim

        self._latent_labels = None
        self.latent_labels = latent_labels

        self._latent_activation = None
        self.latent_activation = latent_activation

        self.latent_track = latent_track

        # class variables
        self.latent_stack = None
        self._latent = None

        self.kwargs = kwargs

        # build model
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
    def latent_labels(self) -> (str, tuple, list, set, None):
        """ label or list of labels of the latent space dimension(s), can be `None` """
        return self._latent_labels

    @latent_labels.setter
    def latent_labels(self, value: (str, tuple, list, set, None)):
        """ label or list of labels of the latent space dimension(s), can be `None` """
        self._latent_labels = value

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

        A stack of dense latent space layers are defined,
        each with their own activation function (which defaults to None).

        The layers are collected in the `Encoder`-instance's `latent_stack` property.

        The layers are added as properties and incrementally labeled via
        'latent_{label_i}' if labels are provided or 'latent_i' otherwise.
        """
        raise NotImplementedError('build')

    def forward(self, x):
        """ forward-method performing the encoding.

        :param x: input tensor
        :returns: latent space tensor (if no labels are defined or if only one latent space dimension is present),
                  or tuple of latent space tensors, one for each latent space dimension.

        The latent space encoding is stored in the `latent` property.
        """
        latent = self.encoder_forward(x)

        if self.latent_track:
            self.set_latent(latent)

        return latent

    def encoder_forward(self, x):
        """ Encoder forward-method performing the encoding.

        :param x: input tensor
        :returns: latent space tensor (if no labels are defined or if only one latent space dimension is present),
                  or tuple of latent space tensors, one for each latent space dimension.
        """
        raise NotImplementedError('encoder_forward')

    @property
    def is_multi_latent(self) -> bool:
        """ Boolean describing whether multiple latent space dimensions are present """
        return self.latent_labels is not None and not isinstance(self.latent_labels, str)

    def set_latent(self, value):
        """ setter for latent space evaluation property

        :param value: tensor whose values are stored in the `latent` property.
        """
        raise NotImplementedError('set_latent')
    @property
    def latent(self) -> (ndarray, {}, {ndarray}):
        """ returns encoding numpy array: x -> z """

        raise NotImplementedError('latent')
