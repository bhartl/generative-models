from __future__ import annotations
import torch.cuda
import torch.nn
from gempy.torch.encoder import Encoder
from gempy.torch.decoder import Decoder
from gempy.torch.auto_encoder import AutoEncoder


class VariationalAutoEncoder(AutoEncoder):
    """ pytorch based Variational Auto Encoder

    Tries to reconstruct an input x -> (mu, log_var) ~ z -> x_hat

    - via an encoding step, predicting the mean (mu) and the confidence (log variance)
      of a datapoint x in the latent space
    - sampling a random number from the normal distribution (mu, std=exp(log_var/2))
    - reconstructing x from z through a decoder

    References:

    - see `deep generative learning <https://www.oreilly.com/library/view/generative-deep-learning/9781492041931/>`_
    - see `towardsdatascience <https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch
    -implementation-3a06bee395ed>`_
    """

    def __init__(self,
                 encoder: (Encoder, dict),
                 decoder: (Decoder, dict),
                 beta: float = 1.,
                 log_scale: float = 0.,
                 reconstruction_loss: (str, callable) = 'mse_reconstruction_loss',
                 ):
        """ Constructs a VAE instance

        :param encoder: Encoder instance or dict, specifying an `AutoEncoder - Encoder` argument
                        (in case of a provided dict, an Encoder instance will be initialized).
        :param decoder: Decoder instance or dict, specifying an `AutoEncoder - Decoder` argument
                        (in case of a provided dict, an Decoder instance will be initialized).
        :param reconstruction_loss: String or callable defining the reconstruction loss (with arguments (x_hat, x),
                                    defaults to 'mse_reconstruction_loss'.
                                    In case of a provided str argument, a VAE-method with the specified name
                                    will be used.
        :param beta: Reconstruction loss weighting factor in the loss function (with respect to the KL divergence).
        :param log_scale: Log scale in case of `gaussian_likelihood' reconstruction loss (defaults to 0.)
        """

        # init and build model:
        super(VariationalAutoEncoder, self).__init__(encoder=encoder, decoder=decoder)

        # check VAE latent space requirements
        assert not isinstance(self.encoder.latent_dim, int), "two dimensional latent shape required"
        assert len(self.encoder.latent_dim) == 2, "two dimensional latent shape required"
        assert self.encoder.latent_dim[0] == self.encoder.latent_dim[1], "two dimensional latent shape required"

        # variables
        self.beta = beta

        # helpers
        self.encoding_sample = None

        self.mu, self.log_var, self.std = None, None, None
        self._zero, _one = None, None

        # for the gaussian likelihood
        self.p, self.q = None, None

        self.reconstruction_loss = getattr(self, reconstruction_loss, reconstruction_loss)

        # training parameter in case of 'gaussian_likelihood' reconstruction_loss
        self.log_scale = None
        self.log_scale = torch.nn.Parameter(torch.Tensor([log_scale]), )
        self.log_scale.requires_grad = False

        # different loss variables
        self.r_loss = None
        self.kl_loss = None
        self.elbo_loss = None

    @property
    def zeros_like(self):
        """ torch zeros tensor with shape of latent space mu dimension """
        return self._zero

    @zeros_like.setter
    def zeros_like(self, value: torch.tensor):
        """ torch zeros tensor with shape of latent space mu dimension """
        try:
            assert self._zero.shape == value.shape

        except (AssertionError, AttributeError):
            self._zero = torch.zeros_like(value)

    @property
    def ones_like(self) -> torch.tensor:
        """ torch ones tensor with shape of latent space log_var dimension """
        return self._one

    @ones_like.setter
    def ones_like(self, value: torch.tensor) -> torch.tensor:
        """ torch ones tensor with shape of latent space log_var dimension """
        try:
            assert self._one.shape == value.shape

        except (AssertionError, AttributeError):
            self._one = torch.ones_like(value)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """ pytorch forward evaluation method

        :param x: input tensors
        :returns: reconstructed inputs with a stochastic sampling step in the latent space
        """

        # encode x to get the mu and variance parameters
        self.encoding = self.encoder(x)
        self.mu, self.log_var = self.encoding
        self.std = torch.exp(self.log_var * 0.5)

        # define the Normal distribution of encoder
        self.q = torch.distributions.Normal(self.mu, self.std)

        # sample z from q
        self.encoding_sample = self.q.rsample()
        return self.decoder(self.encoding_sample)

    def mse_reconstruction_loss(self, x_hat, x):
        """ Mean Square Error - reconstruction loss """
        return torch.nn.MSELoss()(x, x_hat)

    def gaussian_likelihood(self, x_hat, x):
        """ negative Gaussian Likelihood reconstruction loss: -log(p(x|z))

        measure prob of seeing x_hat under p(x|z)
        """
        scale = torch.exp(self.log_scale)
        dist = torch.distributions.Normal(x_hat, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return -log_pxz.sum(dim=(1, 2, 3))

    def kl_divergence(self, z):
        """ Monte Carlo KL divergence

        :param z: sampled encodings
        """

        # define the standardized variational Normal distribution
        self.zeros_like, self.ones_like = self.mu, self.std
        self.p = torch.distributions.Normal(self.zeros_like, self.ones_like)

        # get the probabilities from the equation
        log_qzx = self.q.log_prob(z)
        log_pz = self.p.log_prob(z)

        # kl
        kl = (log_qzx - log_pz)
        kl = kl.sum(-1)

        return kl

    def loss(self, x: torch.tensor, x_hat: torch.tensor) -> torch.tensor:
        """ total loss function of the VAE

        consists of `mean(KL_loss + beta * r_loss)`
        """

        # reconstruction
        r_loss = self.reconstruction_loss(x_hat=x_hat, x=x)

        # kl
        kl_loss = self.kl_divergence(z=self.encoding_sample)

        # elbo
        self.kl_loss = kl_loss.mean()
        self.r_loss = r_loss.mean()
        self.elbo_loss = (self.kl_loss + self.beta * self.r_loss).mean()

        return self.elbo_loss

    def training_step(self, x):
        """ VAE forward and loss evaluation on input x

        :param x: torch input tensor
        :returns: tuple of (x_hat, loss), i.e., reconstructed input x and corresponding loss function
        """
        x_hat = self(x)  # get decoding
        loss = self.loss(x=x, x_hat=x_hat)
        return x_hat, loss


if __name__ == '__main__':
    from gempy.torch.encoder import ConvEncoder
    from gempy.torch.decoder import ConvTDecoder

    input_dim = (1, 28, 28)
    z_dim = (2, 2)

    cnn_encoder = ConvEncoder(
        input_shape=input_dim,
        filters=(32, 64, 64, 64),
        kernels_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        activation='leaky_relu',
        latent_dim=z_dim,
        latent_labels=('mu', 'log_var'),  # None,  #
        latent_activation=None,
        latent_track=True
    )

    cnn_decoder = ConvTDecoder(
        latent_dim=z_dim[0],
        latent_upscale=(64, 7, 7),
        filters=[64, 64, 32, 1],
        kernels_size=[3, 4, 4, 3],
        strides=[1, 2, 2, 1],
        activation=['leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid'],
        latent_merge=False,
        latent_activation=None,
    )

    cnn_vae = VariationalAutoEncoder(encoder=cnn_encoder, decoder=cnn_decoder, beta=1.)

    print(cnn_vae)
    print('input shape     :', cnn_vae.encoder.conv_stack_shape_in)
    print('latent shape    :', cnn_vae.encoder.latent_dim)
    print('output shape    :', cnn_vae.decoder.conv_transpose_stack_shape_out)

    x_random = torch.randn(10, *input_dim)
    y, loss = cnn_vae.training_step(x_random)

    try:
        print('latent space    :', {k: v.shape for k, v in cnn_encoder.latent_torch.items()})
    except AttributeError:
        print('latent space    :', tuple(v.shape for v in cnn_encoder.latent_torch))

    print('output shape    :', y.shape)
    print('elbo loss       :', loss)
