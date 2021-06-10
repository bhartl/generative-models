from __future__ import annotations
import torch.cuda
import torch.nn
from gempy.torch.encoder import Encoder
from gempy.torch.decoder import Decoder
from gempy.torch.auto_encoder import AutoEncoder


class VariationalAutoEncoder(AutoEncoder):
    """ pytorch based Auto Encoder

    see `towardsdatascience https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch
    -implementation-3a06bee395ed`_
    """

    def __init__(self, encoder=Encoder, decoder=Decoder, beta=1., log_scale=0.):
        # define model:
        super(VariationalAutoEncoder, self).__init__(encoder=encoder, decoder=decoder)

        assert not isinstance(self.encoder.latent_shape, int), "two dimensional latent shape required"
        assert len(self.encoder.latent_shape) == 2, "two dimensional latent shape required"
        assert self.encoder.latent_shape[0] == self.encoder.latent_shape[1], "two dimensional latent shape required"

        # training parameter
        self.log_scale = None

        # variables
        self.beta = beta

        # helpers
        self.encoding_sample = None

        self.mu, self.log_var, self.std = None, None, None
        self._zero, _one = None, None

        # for the gaussian likelihood
        self.p, self.q = None, None

        self.log_scale = torch.nn.Parameter(torch.Tensor([log_scale]), )
        self.log_scale.requires_grad = False

    @property
    def zeros_like(self):
        return self._zero

    @zeros_like.setter
    def zeros_like(self, value):
        try:
            assert self._zero.shape == value.shape

        except (AssertionError, AttributeError):
            self._zero = torch.zeros_like(value)

    @property
    def ones_like(self):
        return self._one

    @ones_like.setter
    def ones_like(self, value):
        try:
            assert self._one.shape == value.shape

        except (AssertionError, AttributeError):
            self._one = torch.ones_like(value)

    def forward(self, x):

        # encode x to get the mu and variance parameters
        self.encoding = self.encoder(x)
        self.mu, self.log_var = self.encoding
        self.std = torch.exp(self.log_var * 0.5)

        # define the firstNormal distribution of encoder
        self.q = torch.distributions.Normal(self.mu, self.std)

        # sample z from q
        self.encoding_sample = self.q.rsample()
        return self.decoder(self.encoding_sample)

    def gaussian_likelihood(self, x_hat, x):
        scale = torch.exp(self.log_scale)
        dist = torch.distributions.Normal(x_hat, scale)

        # measure prob of seeing image under p(x|z)
        log_pxz = dist.log_prob(x)
        return log_pxz.sum(dim=(1, 2, 3))

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

    def elbo_loss(self, x, x_hat):
        # reconstruction
        recon_loss = self.gaussian_likelihood(x_hat=x_hat, x=x)

        # kl
        kl = self.kl_divergence(z=self.encoding_sample)

        # elbo
        elbo = (kl - self.beta * recon_loss)
        elbo = elbo.mean()

        return elbo

    def training_step(self, x):
        x_hat = self(x)  # get decoding
        loss = self.elbo_loss(x=x, x_hat=x_hat)
        return x_hat, loss


if __name__ == '__main__':
    from gempy.torch.encoder import ConvEncoder
    from gempy.torch.decoder import ConvDecoder

    input_dim = (1, 28, 28)
    z_dim = (2, 2)

    cnn_encoder = ConvEncoder(
        input_shape=input_dim,
        filters=(32, 64, 64, 64),
        kernels_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        activation='leaky_relu',
        latent_shape=z_dim,
        latent_labels=('mu', 'log_var'),
        latent_activation=None,
    )

    cnn_decoder = ConvDecoder(
        latent_shape=z_dim[0],
        latent_upscale=(64, 7, 7),
        filters=[64, 64, 32, 1],
        kernels_size=[3, 4, 4, 3],
        strides=[1, 2, 2, 1],
        activation=['leaky_relu', 'leaky_relu', 'leaky_relu', 'sigmoid'],
        latent_merge=False,
        latent_activation=None,
    )

    cnn_vae = VariationalAutoEncoder(
        encoder=cnn_encoder,
        decoder=cnn_decoder,
        beta=1.
    )

    print(cnn_vae)
    print('input shape     :', cnn_vae.encoder.conv_stack_shape_in)
    print('latent shape    :', cnn_vae.encoder.latent_shape)
    print('output shape    :', cnn_vae.decoder.conv_stack_shape_out)

    x_random = torch.randn(1, *input_dim)
    y, loss = cnn_vae.training_step(x_random)

    print('latent space    :', cnn_encoder.latent_torch)
    print('output shape    :', y.shape)
    print('elbo loss       :', loss)
