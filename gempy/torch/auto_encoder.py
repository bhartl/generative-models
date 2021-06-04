from __future__ import annotations
import torch.cuda
import torch.nn
from . import Encoder, Decoder

class AutoEncoder(torch.nn.Module):
    """ pytorch based Auto Encoder """

    def __init__(self, encoder=Encoder, decoder=Decoder):
        super(AutoEncoder, self).__init__()

        self._encoder = None
        self.encoder = encoder

        self._decoder = None
        self.decoder = decoder

    @property
    def encoder(self) -> Encoder:
        return self._encoder

    @encoder.setter
    def encoder(self, value: (Encoder, dict)):
        if isinstance(value, dict):
            self._encoder = Encoder(**value)
            return

        assert isinstance(value, Encoder)
        self._encoder = value

    @property
    def decoder(self) -> Decoder:
        return self._decoder

    @decoder.setter
    def decoder(self, value: (Decoder, dict)):
        if isinstance(value, dict):
            self._decoder = Decoder(**value)
            return

        assert isinstance(value, Decoder)
        self._decoder = value

    def forward(self, x):
        self.encoder(x)
        return self.decoder(self.encoding)

    def accelerate(self, **kwargs) -> AutoEncoder:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using {} device".format(device))
        return self.to(device, **kwargs)


class VariationalAutoEncoder(AutoEncoder):
    """ pytorch based variational auto encoder """

    def __init__(self, encoder=Encoder, decoder=Decoder):
        super(VariationalAutoEncoder, self).__init__()