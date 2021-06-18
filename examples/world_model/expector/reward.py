import torch


class BinaryRewardModel(torch.nn.Module):
    def __init__(self, input_size=35, reward_size=1):
        super(BinaryRewardModel, self).__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_size, reward_size),
        )

    def forward(self, x):
        y = self.network(x)
        y[y < 0] = 0.
        y[y > 0] = 1.
        return y


