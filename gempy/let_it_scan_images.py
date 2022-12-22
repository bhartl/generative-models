from torch import tensor, reshape, cat, stack, split
from torch import nn, ones, zeros, no_grad, randn_like
from numpy import prod
from torchvision.transforms import Pad
from typing import Union
from gempy.torch_.mdn import MixtureDensityNetwork


class Lisi(nn.Module):
    """ **L**et **I**t **S**Scan **I**mages (LISI) """

    def __init__(self,
                 in_channels: int = 1,
                 kernel_size: int = 5,
                 embedding_size: int = 32,
                 dropout: float = 0.1,
                 hidden_size: int = 32,
                 n_components=5,
                 head=None,
                 padding_mode='constant',
                 noise_level=0.01,
                 max_iter: Union[int, float] = 1,
                 ):
        """  Constructs a `Lisi` instance """

        nn.Module.__init__(self)

        # hyperparams
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_components = n_components

        self.max_iter = max_iter
        self.padding_mode = padding_mode

        self.noise_level = noise_level

        self.pad = None
        self.feature_embedding = None
        self.positional_embedding = None
        self.sequence_module = None
        self.mdn_module = None
        self.confidence_model = None

        self._build()
        self.head = head

    def _build(self, ):
        self.pad = Pad(self.kernel_size//2, padding_mode=self.padding_mode)
        self.feature_embedding = nn.Sequential(nn.Conv2d(self.in_channels, self.embedding_size, self.kernel_size),
                                               nn.Dropout(self.dropout),
                                               nn.Flatten(),
                                               nn.BatchNorm1d(self.embedding_size),
                                               nn.ReLU(),
                                               )

        self.positional_embedding = nn.Linear(2, self.embedding_size)

        self.sequence_module = nn.LSTM(self.embedding_size,
                                       self.hidden_size,
                                       num_layers=1,
                                       batch_first=True,
                                       )

        self.mdn_module = MixtureDensityNetwork(self.hidden_size,
                                                sample_size=self.embedding_size + 2,
                                                n_components=self.n_components,
                                                forward_mode=MixtureDensityNetwork.FORWARD_SAMPLE
                                                )

    def to(self, device):
        nn.Module.to(self, device)

        self.pad = self.pad.to(device)
        self.feature_embedding = self.feature_embedding.to(device)
        self.positional_embedding = self.positional_embedding.to(device)
        self.sequence_module = self.sequence_module.to(device)
        self.mdn_module = self.mdn_module.to(device)

        return self

    def forward(self, x, xy=None, track_focus=True):
        if xy is None:
            xy = ones(size=(len(x), 2), device=x.device) * 0.5

        x = self.pad(x)

        i = 0
        done = False
        max_iter = int((prod(x.shape[-2:])/self.kernel_size/self.kernel_size)*self.max_iter)

        embeddings, coords, pred_embeddings = [], [], []
        x_context, x_state, pred_embedding = None, None, None

        while not done:
            patch = self.get_patch(x, xy)
            patch = self.augment(patch)
            embedding = self.feature_embedding(patch) + self.positional_embedding(xy)

            if track_focus:
                coords.append(xy)
                embeddings.append(embedding)

            if pred_embedding is not None and track_focus:
                    pred_embeddings.append(pred_embedding)

            i += 1
            done |= (i >= max_iter)

            if not done or x_context is None:
                x_context, x_state = self.sequence_module(embedding.unsqueeze(dim=1), x_state)
                pred_embedding, xy = self.mixture_density_model(x_state[0].squeeze(0))  # x_state -> (hidden_state, cell_state)

        x_context = x_context.squeeze(dim=1)
        y = self.head(x_context) if self.head is not None else x_context

        if not track_focus:
            return y

        return y, (coords, stack(embeddings, dim=1), stack(pred_embeddings, dim=1))

    def get_patch(self, x, xy):
        xy_start = (xy * (tensor(x.shape[-2:], device=x.device) - 1) - self.kernel_size//2).long()
        xy_end = (xy_start + self.kernel_size).long()
        patches = [x[i, :, xy_start[i, 0]:xy_end[i, 0], xy_start[i, 1]:xy_end[i, 1]] for i in range(len(x))]
        return stack(patches).clone().detach().requires_grad_(True)

    def augment(self, x):
        x = self.augment_noise(x)
        x = self.augment_focus(x)
        return x

    def augment_noise(self, x):
        """ add noise to the patch"""
        return x + randn_like(x) * self.noise_level

    def augment_focus(self, x):
        """ apply radial blurring filter, augmenting focus of eye-sight (TODO) """
        return x

    def mixture_density_model(self, x):
        x = self.mdn_module(x)
        embedding, xy = split(x, self.embedding_size, dim=1)
        xy = nn.Sigmoid()(xy)
        return embedding, xy


if __name__ == '__main__':
    from torch import randn

    img_size = (1, 32, 32)

    kernel_size = 3
    embedding_size = 8
    device = 'cuda'  # 'cpu

    x = randn(size=(9, *img_size), device=device)

    lisi = Lisi(in_channels=img_size[0],
                kernel_size=kernel_size,
                embedding_size=embedding_size,
                max_iter=1.,
                )
    lisi = lisi.to(device)

    y, (coords, patches, pred_embeddings)  = lisi(x)

    import matplotlib.pyplot as plt
    coords = stack(coords, dim=1)

    if not device == 'cpu':
        x = x.cpu()
        coords = coords.cpu()

    f, axes = plt.subplots(3, 3, sharex=True, sharey=True)
    for i in range(3):
        for j in range(3):
            ij = i*3 + j

            axes[i, j].imshow(x[ij].reshape(tuple(reversed(img_size))))
            axes[i, j].plot(coords[ij, :, 0] * img_size[1], coords[ij, :, 1] * img_size[2],
                            markersize=1,
                            linewidth=3,
                            color='black',
                            alpha=0.6,
                            label=f'batch {ij}')

            if j == 0:
                axes[i, 0].set_ylabel('y')

            if i == 2:
                axes[2, j].set_xlabel('x')

    plt.show()
