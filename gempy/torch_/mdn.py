""" mdn-code from `tonyduan <https://github.com/tonyduan/mdn>`_ under the MIT-License """

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.distributions import Normal, OneHotCategorical, RelaxedOneHotCategorical, Uniform


class MixtureDensityNetwork(pl.LightningModule):
    """
    Mixture density wired_rnn.
    [ Bishop, 1994 ]
    Parameters
    ----------
    input_size: int; dimensionality of the covariates
    sample_size: int; dimensionality of the response variable
    n_components: int; number of components in the mixture model
    """

    FORWARD_SAMPLE = 'sample'
    FORWARD_PARAMS = 'params'

    def __init__(self, input_size, sample_size, sample_shape=None, n_components=1,
                 beta=0., learning_rate=0.005,
                 hidden_pi=None, hidden_normal=None, forward_mode=FORWARD_SAMPLE):

        super().__init__()

        # parameters
        self.input_size = input_size
        self.sample_size = sample_size
        self.sample_shape = sample_shape
        self.n_components = n_components
        self.learning_rate = learning_rate

        self.hidden_pi = hidden_pi
        self.hidden_normal = hidden_normal

        # wired_rnn
        self.pi_network = CategoricalNetwork(input_size, n_components, beta=beta, hidden_dim=hidden_pi)
        self.normal_network = MixtureDiagNormalNetwork(input_size, sample_size, n_components, hidden_dim=hidden_normal)

        assert forward_mode in (MixtureDensityNetwork.FORWARD_SAMPLE, MixtureDensityNetwork.FORWARD_PARAMS)
        self.forward_mode = forward_mode

    def to(self, device='cpu'):
        self.pi_network = self.pi_network.to(device)
        self.normal_network = self.normal_network.to(device)
        return self

    def to_dict(self):
        dict_repr = dict(
            input_size=self.input_size,
            sample_size=self.sample_size,
            sample_shape=list(self.sample_shape) if self.sample_shape is not None else None,
            n_components=self.n_components,
            beta=self.beta,
            learning_rate=self.learning_rate,
            hidden_pi=self.hidden_pi,
            hidden_normal=self.hidden_normal,
            forward_mode=self.forward_mode,
        )

        return dict_repr

    @property
    def beta(self):
        return self.pi_network.beta

    @beta.setter
    def beta(self, value):
        self.pi_network.beta = value

    def forward_gaussian_mixture_parameters(self, x):
        pi = self.pi_network(x)
        normal = self.normal_network(x)
        return pi, normal

    def forward(self, x):
        if self.forward_mode == 'sample':
            return self.sample(x, self.beta)
        return self.forward_gaussian_mixture_parameters(x)

    def loss(self, x, y):
        pi, normal = self.forward_gaussian_mixture_parameters(x)
        return self.max_loglik(pi, normal, y)

    @staticmethod
    def max_loglik(pi, normal, y):

        if len(y.shape) == len(normal.loc.shape):  # [BS, SEQ, FEATURES] == [BS x SEQ, N_COMPONENTS, FEATURES], with or without SEQ possible
            y = y.reshape(y.shape[0] * y.shape[1], -1)

        loglik = normal.log_prob(y.unsqueeze(1).expand_as(normal.loc))
        loglik = torch.sum(loglik, dim=2)
        loss = -torch.logsumexp(torch.log(pi.probs) + loglik, dim=1)
        return loss

    def sample(self, x, beta=None):
        self.beta = beta
        pi, normal = self.forward_gaussian_mixture_parameters(x)
        samples = torch.sum(pi.sample().unsqueeze(2) * normal.sample(), dim=1)

        if self.sample_shape is not None:
            samples = samples.view((*samples.shape[:-1], *self.sample_shape))

        return samples

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def training_step(self, batch, batch_idx=None):
        x, y = batch
        loss = self.loss(x, y).mean()
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x, y = batch
        loss = self.loss(x, y).mean()
        self.log('val_loss', loss)
        return loss

    def fit(self, train_dataloader, val_dataloader=None, max_epochs=100, model_path='examples/models/world_model/mdn/',
            val_interval=1):

        print('initialize checkpoints')
        checkpoint_callback_loss = ModelCheckpoint(monitor='loss', save_top_k=1, mode='min')
        callbacks = [checkpoint_callback_loss]

        if val_dataloader is not None:
            checkpoint_callback_val_loss = ModelCheckpoint(monitor='val_loss')
            callbacks.append(checkpoint_callback_val_loss)

        print('initialize trainer')
        trainer = Trainer(max_epochs=max_epochs,
                          gpus=int(torch.cuda.is_available()),
                          default_root_dir=model_path,
                          callbacks=callbacks,
                          precision=16,  # accelerate
                          check_val_every_n_epoch=val_interval,
                          )
        print('start training')
        dataloaders = [train_dataloader,]

        if val_dataloader is not None:
            dataloaders.append(val_dataloader)
        trainer.fit(self, *dataloaders)

        print('done')
        return self

    def fit_direct(self, x, y, max_epochs=100):

        print('start training')
        optimizer = self.configure_optimizers()

        for i in range(max_epochs):
            optimizer.zero_grad()
            loss = self.training_step((x, y))
            loss.backward()
            optimizer.step()
            if i % 100 == 0:
                print(f"Iter: {i}\t" + f"Loss: {loss.data:.2f}")


class MixtureDiagNormalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, n_components, hidden_dim=None):
        super().__init__()

        self.n_components = n_components

        layer_stack = []
        if hidden_dim is not None:
            if isinstance(hidden_dim, int):
                hidden_dim = [hidden_dim]

            for i in range(len(hidden_dim)):
                hidden_in_dim = hidden_dim[i-1] if i > 0 else in_dim
                hidden_dim_out = hidden_dim[i]
                in_dim = hidden_dim_out

                layer_stack.append(nn.Linear(hidden_in_dim, hidden_dim_out))
                layer_stack.append(nn.ELU())

        layer_stack.append(nn.Linear(in_dim, 2 * out_dim * n_components))
        self.network = nn.Sequential(*layer_stack)

    def forward(self, x):
        params = self.network(x)
        mean, log_var = torch.split(params, params.shape[1] // 2, dim=1)
        mean = torch.stack(mean.split(mean.shape[1] // self.n_components, 1))
        log_var = torch.stack(log_var.split(log_var.shape[1] // self.n_components, 1))
        return Normal(mean.transpose(0, 1), torch.exp(log_var * 0.5).transpose(0, 1),)


class CategoricalNetwork(nn.Module):

    def __init__(self, in_dim, out_dim, hidden_dim=None, beta=None):
        super().__init__()

        self.beta = beta

        layer_stack = []
        if hidden_dim is not None:
            if isinstance(hidden_dim, int):
                hidden_dim = [hidden_dim]

            for i in range(len(hidden_dim)):
                hidden_in_dim = hidden_dim[i-1] if i > 0 else in_dim
                hidden_dim_out = hidden_dim[i]
                in_dim = hidden_dim_out

                layer_stack.append(nn.Linear(hidden_in_dim, hidden_dim_out))
                layer_stack.append(nn.ELU())

        layer_stack.append(nn.Linear(in_dim, out_dim))
        self.network = nn.Sequential(*layer_stack)

    def forward(self, x):
        params = self.network(x)

        if self.beta not in (None, 0., np.inf):
            RelaxedOneHotCategorical(temperature=1./self.beta, logits=params)

        return OneHotCategorical(logits=params)


def ex_1d(n_iterations: int = 2000, datapoints=512, batch_size=512, workers=4, gpu=False, beta=None):
    """example application of the Mixture Density Network"""

    import matplotlib.pyplot as plt
    from torch.utils.data import TensorDataset, DataLoader

    def gen_data(n=datapoints):
        y = np.linspace(-1, 1, n)
        x = 7 * np.sin(5 * y) + 0.5 * y + 0.5 * np.random.randn(*y.shape)
        return x[:, np.newaxis], y[:, np.newaxis]

    def plot_data(x, y):
        plt.hist2d(x, y, bins=35)
        plt.xlim(-8, 8)
        plt.ylim(-1, 1)
        plt.axis('off')

    x, y = gen_data()
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    model = MixtureDensityNetwork(input_size=1,
                                  sample_size=1,
                                  n_components=3,
                                  learning_rate=5e-3,
                                  hidden_pi=[5, ],
                                  hidden_normal=[5, ],
                                  beta=beta
                                  )

    if gpu:
        dataset = TensorDataset(x, y)
        dataloader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, num_workers=workers,
                                                 pin_memory=True, shuffle=True)

        model.fit(max_epochs=n_iterations, train_dataloader=dataloader, val_interval=100)

    else:
        model.fit_direct(x, y, max_epochs=n_iterations)

    samples = model.sample(x)
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plot_data(x[:, 0].numpy(), y[:, 0].numpy())
    plt.title("Observed data")
    plt.subplot(1, 2, 2)
    plot_data(x[:, 0].numpy(), samples[:, 0].numpy())
    plt.title("Sampled data")
    plt.show()


if __name__ == '__main__':
    import argh

    argh.dispatch_commands([
        ex_1d,
    ])
