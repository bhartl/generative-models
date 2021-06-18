import torch
from examples.world_model.expector.lstm_helper import build_lstm, build_ln_lstm


class LSTMModel(torch.nn.Module):
    def __init__(self,
                 latent_size=32, action_size=3, reward_size=1,
                 hidden_size=256, num_layers=1, dropout=0.25, use_layer_norm=True,
                 ):
        super(LSTMModel, self).__init__()

        # network parameters
        self.input_size = latent_size + action_size + reward_size
        self.latent_size = latent_size
        self.action_size = action_size
        self.reward_size = reward_size

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = dropout
        self.use_layer_norm = use_layer_norm

        # actual lstm model
        self.rnn = self._build(self.input_size, hidden_size, num_layers, dropout)

        # states
        self.output = None
        self._states = None
        self.hidden_state = None
        self.cell_state = None

    def _build(self, input_size, hidden_size, num_layers, dropout):
        if self.use_layer_norm:
            return build_ln_lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

        return build_lstm(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                          batch_first=False, bidirectional=False, )

    @property
    def states(self):
        if self._states is None:
            # initialize state randomly
            self.states = None

        return self._states

    @states.setter
    def states(self, value):
        if value is None:
            self._states = [(torch.randn((1, self.hidden_size)),  # initial hidden state per layer
                             torch.randn((1, self.hidden_size)))  # initial cell state per layer
                            for __ in range(self.num_layers)
                           ]
        else:
            self._states = value

        self.hidden_state, self.cell_state = self._states[-1]

    def forward(self, x: (tuple, torch.Tensor), states=None):
        """
        :param x: concatenated [latent, action, reward] inputs (concatenated in last dimension)
        :param states: recurrent states (hidden, cell) of lstm layer
        """
        if isinstance(x, tuple):
            x = torch.cat(x, -1)
        assert isinstance(x, torch.Tensor)

        if states is None:
            states = self.states

        self.output, self.states = self.rnn(x, states)
        return self.output, self.states

    def predict(self, x_latent, x_action, x_reward):
        x = torch.cat((x_latent, x_action, x_reward), -1)
        return self.forward(x, self.states)


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

    model = LSTMModel(latent_size=latent_size, action_size=action_size, reward_size=reward_size,
                      hidden_size=hidden_size, num_layers=num_layers)

    rnn_output, rnn_states = model.forward((latent, action, reward))  # generates random initial state
    rnn_output, rnn_states = model.forward((latent, action, reward), states=rnn_states)
    model.predict(latent, action, reward)

    print("output shape:", rnn_output.shape)
    print("state shape after forward:", [tuple(ti.shape for ti in layer_state) for layer_state in rnn_states])
    print("state shape after predict:", [tuple(ti.shape for ti in layer_state) for layer_state in model.states])
    print("hidden shape after predict:", model.hidden_state.shape)
    print("cell shape after predict:", model.cell_state.shape)

