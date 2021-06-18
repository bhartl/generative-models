from __future__ import annotations
import torch.nn
import torch.nn.functional as F
import pytorch_lightning as pl
from gempy.torch.encoder import Encoder
from gempy.torch.decoder import Decoder


class AutoEncoder(pl.LightningModule):

    """ pytorch based auto encoder: x (feature space) -> z (latent space) -> x_hat (reconstructed)

    Usage and Inheritance:

    - Inherits from `torch.nn.Module`
    - `_build` routine is to be overwritten by more specific `AutoEncoder` child-classes to define the network
    - `_torch_forward` routine is to be overwritten by more specific `AutoEncoder` child-classes
      to define the tensor flow

    Comments:

    - multiple latent space layers/dimensions are possible
    - retrieval of evaluations possible via dictionary mapping (if labels are provided)
    """

    def __init__(self,
                 encoder: (Encoder, dict),
                 decoder: (Decoder, dict),
                 learning_rate: float = 1e-3,
                 batch_size: int = 1,
                 ):
        """ Constructs an `AutoEncoder` instance

        :param encoder: Encoder instance or dict from which an Encoder instance can be constructed
        :param decoder: Decoder instance or dict from which an Decoder instance can be constructed
        :param learning_rate: Learning rate (float, defaults to 1e-3)
        :param batch_size: Batch-size (int, defaults to 1)
        """
        super(AutoEncoder, self).__init__()

        self.encoder = None
        self.set_encoder(encoder)

        self.encoding = None

        self.decoder = None
        self.set_decoder(decoder)

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self._train_dataloader = None
        self._val_dataloader = None

    def set_encoder(self, value: (Encoder, dict)):
        """ defines the Encoder of the AutoEncoder

        An `encoder` property is set, specifying the Encoder torch model in the AutoEncoder.

        :param value: Encoder instance or dict-representation thereof
        """
        if isinstance(value, dict):
            self.encoder = Encoder(**value)
            return

        assert isinstance(value, Encoder)
        self.encoder = value

    def set_decoder(self, value: (Decoder, dict)):
        """ defines the Decoder of the AutoEncoder

        An `decoder` property is set, specifying the Decoder torch model in the AutoEncoder.

        :param value: Decoder instance or dict-representation thereof
        """
        if isinstance(value, dict):
            self.decoder = Decoder(**value)
            return

        assert isinstance(value, Decoder)
        self.decoder = value

    def forward(self, x):
        """ torch forward method, reconstructing the input x after compressing it through a bottle-neck latent space

        If the encoder is setup accordingly (via the `latent_track`) switch, the latent-space encoding can be retrieved
        via the `encoder.latent_torch` or `encoder.latent` properties.

        :param x: torch input tensor
        :returns: reconstruction x_hat of the input
        """
        self.encoding = self.encoder(x)
        return self.decoder(*self.encoding)

    def training_step(self, batch, batch_idx=None):
        # training_step defines the train loop. It is independent of forward
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx=None):
        # validation_step defines the train loop. It is independent of forward
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return self._train_dataloader

    def set_train_dataloader(self, value):
        self._train_dataloader = value

    def val_dataloader(self):
        return self._val_dataloader

    def set_val_dataloader(self, value):
        self._val_dataloader = value


if __name__ == '__main__':
    from gempy.torch.encoder import ConvEncoder
    from gempy.torch.decoder import ConvTDecoder

    input_dim = (1, 28, 28)
    z_dim = (2, 3, 4)

    cnn_encoder = ConvEncoder(
        input_shape=input_dim,
        filters=(32, 64, 64, 64),
        kernels_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        activation='leaky_relu',
        latent_dim=z_dim,
        latent_labels=('z', 'mu', 'a'),
        latent_activation='sigmoid',
        latent_track=True
    )

    cnn_decoder = ConvTDecoder(
        latent_dim=z_dim,
        latent_upscale=cnn_encoder.conv_stack_shape_out,
        filters=[64, 64, 32, 1],
        kernels_size=[3, 4, 4, 3],
        strides=[1, 2, 2, 1],
        activation='leaky_relu',
        latent_merge=False,
        latent_activation=None,
    )

    cnn_auto_encoder = AutoEncoder(
        encoder=cnn_encoder,
        decoder=cnn_decoder,
    )

    print(cnn_auto_encoder)
    print('input shape     :', cnn_auto_encoder.encoder.conv_stack_shape_in)
    print('latent shape    :', z_dim)
    print('output shape    :', cnn_auto_encoder.decoder.deconv_stack_shape_out)

    x_random = torch.randn(1, *input_dim)
    y = cnn_auto_encoder(x_random)

    print('latent space    :', cnn_encoder.latent_torch)
    print('output shape    :', y.shape)

