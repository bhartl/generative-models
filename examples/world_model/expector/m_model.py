""" based on https://github.com/hardmaru/WorldModelsExperiments/blob/master/carracing/rnn/rnn.py """

import pytorch_lightning as pl
from examples.world_model.expector.lstm import LSTMModel
from examples.world_model.expector.mdn import MixtureDensityNetwork
from examples.world_model.expector.reward import BinaryRewardModel
import torch
import torch.nn.functional as F


class MModel(pl.LightningModule):

    def __init__(self,
                 rnn: (LSTMModel, dict) = None,
                 mdn: (MixtureDensityNetwork, dict) = None,
                 reward: (BinaryRewardModel, dict) = None,
                 learning_rate: float = 1e-3,
                 batch_size: int = 1,
                 r_loss_factor: float = 1.,
                 ):
        """ Constructs an `MModel` instance

        :param rnn: Recurrent Neural Network model
        :param mdn: Mixture Denstiy Neural Network
        :param reward: Reward model
        :param learning_rate: Learning rate (float, defaults to 1e-3)
        :param batch_size: Batch-size (int, defaults to 1)
        :param r_loss_factor: Multiplier to the reward-loss (as compared to the z-loss)
        """

        super(MModel, self).__init__()

        self.rnn_model = None
        self.mdn_model = None
        self.reward_model = None
        self._build(rnn, mdn, reward)

        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.rnn_output = None
        self.expected_latent = None
        self.expected_reward = None
        self.r_loss_factor = r_loss_factor

    def _build(self, rnn, mdn, reward):
        self.rnn_model = self._build_rnn(rnn)
        self.mdn_model = self._build_mdn(mdn)
        self.reward_model = self._build_reward(reward)

    def _build_rnn(self, rnn_model) -> LSTMModel:
        if isinstance(rnn_model, LSTMModel):
            return rnn_model

        if rnn_model is None:
            rnn_model = {}

        assert isinstance(rnn_model, dict)
        rnn_model['latent_size'] = rnn_model.get('latent_size', 32)
        rnn_model['action_size'] = rnn_model.get('action_size', 3)
        rnn_model['reward_size'] = rnn_model.get('reward_size', 1)
        rnn_model['hidden_size'] = rnn_model.get('hidden_size', 256)
        rnn_model['num_layers'] = rnn_model.get('num_layers', 1)

        return LSTMModel(**rnn_model)

    def _build_mdn(self, mdn_model) -> MixtureDensityNetwork:
        if isinstance(mdn_model, MixtureDensityNetwork):
            assert mdn_model.input_size == self.rnn_model.input_size
            return mdn_model

        if mdn_model is None:
            mdn_model = {}

        assert isinstance(mdn_model, dict)
        mdn_model['input_size'] = mdn_model.get('input_size', self.rnn_model.input_size)
        mdn_model['sample_size'] = mdn_model.get('sample_size', self.rnn_model.latent_size)
        mdn_model['n_components'] = mdn_model.get('n_components', 5)

        return MixtureDensityNetwork(**mdn_model)

    def _build_reward(self, reward_model) -> BinaryRewardModel:
        if isinstance(reward_model, BinaryRewardModel):
            return reward_model

        if reward_model is None:
            reward_model = {}

        assert isinstance(reward_model, dict)
        reward_model['input_size'] = reward_model.get('input_size', self.rnn_model.input_size)
        reward_model['reward_size'] = reward_model.get('reward_size', 1)

        return BinaryRewardModel(**reward_model)

    def forward(self, latent, action, reward):
        self.rnn_output, _ = self.rnn_model.forward((latent, action, reward), self.rnn_model.states)
        self.expected_latent = self.mdn_model.forward(self.rnn_output)
        self.expected_reward = self.reward_model(self.rnn_output)

        return self.expected_latent, self.expected_reward

    def latent_loss(self, x, y) -> torch.Tensor:
        pass

    def reward_loss(self, x, y) -> torch.Tensor:
        pass

    def loss(self, latent, reward, expected_latent, expected_reward) -> torch.Tensor:
        latent_loss = self.latent_loss(expected_latent, latent)
        reward_loss = self.reward_loss(expected_reward, reward)
        return (latent_loss + self.r_loss_factor * reward_loss).mean()


if __name__ == '__main__':
    sequence_len = 5
    batch_size = 10

    latent_size = 32
    action_size = 3
    reward_size = 1

    hidden_size = 256
    num_layers = 1

    latent = torch.randn((sequence_len, batch_size, latent_size))
    action = torch.randn((sequence_len, batch_size, action_size))
    reward = torch.randn((sequence_len, batch_size, reward_size))

    print("latent shape", latent.shape)
    print("action shape", action.shape)
    print("reward shape", reward.shape)

    model = MModel()

    expected_latent, expected_reward = model.forward(latent, action, reward)

    print("expected_latent shape:", expected_latent.shape)
    print("expected_reward shape:", expected_reward.shape)
