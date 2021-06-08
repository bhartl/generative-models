from __future__ import annotations
import torch.cuda
import torch.nn
from gempy.torch.encoder import Encoder
from gempy.torch.decoder import Decoder


class AutoEncoder(torch.nn.Module):
    """ pytorch based Auto Encoder """

    def __init__(self, encoder=Encoder, decoder=Decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = None
        self.set_encoder(encoder)

        self.encoding = None

        self.decoder = None
        self.set_decoder(decoder)

    def set_encoder(self, value: (Encoder, dict)):
        if isinstance(value, dict):
            self.encoder = Encoder(**value)
            return

        assert isinstance(value, Encoder)
        self.encoder = value

    def set_decoder(self, value: (Decoder, dict)):
        if isinstance(value, dict):
            self.decoder = Decoder(**value)
            return

        assert isinstance(value, Decoder)
        self.decoder = value

    def forward(self, x):
        self.encoding = self.encoder(x)
        return self.decoder(*self.encoding)

    def accelerate(self, **kwargs) -> 'AutoEncoder':
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        return self.to(device, **kwargs)


if __name__ == '__main__':
    from gempy.torch.encoder import ConvEncoder
    from gempy.torch.decoder import ConvDecoder

    input_dim = (1, 28, 28)
    z_dim = (2, 2)

    cnn_encoder = ConvEncoder(
        input_shape=input_dim,
        filters=(32, 64, 64, 64),
        kernel_size=(3, 3, 3, 3),
        strides=(1, 2, 2, 1),
        activation='leaky_relu',
        latent_dim=z_dim,
        latent_labels=('z', 'mu', 'a'),
        latent_activation='sigmoid',
    )

    cnn_decoder = ConvDecoder(
        latent_shape=z_dim,
        latent_upscale=(64, 3, 3),
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
    print('output shape    :', cnn_auto_encoder.decoder.conv_stack_shape_out)

    x_random = torch.randn(1, *input_dim)
    y = cnn_auto_encoder(x_random)

    print('latent space    :', cnn_encoder.latent_torch)
    print('output shape    :', y.shape)

