#pylint: disable=no-member
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn

from .utils import fanin_init

# Numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Use the values as observed in Rlkit
B_INIT_VALUE = 0.1
W_INIT_VALUE = 3e-3


class QModel(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_size: int = 256):
        super(QModel, self).__init__()

        self.recurrent_layer = nn.LSTM(
            num_inputs, hidden_size, batch_first=True)

        self.layer1 = nn.Linear(hidden_size + num_actions, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, 1)

        for layer in [self.layer1, self.layer2, self.layer3]:
            layer.bias.data.fill_(B_INIT_VALUE)

            # fanin init
            fanin_init(layer.weight)

    def forward(self, state, action):
        # Recurrent layers
        _, (state, _) = self.recurrent_layer(state)
        state = state.squeeze(0)

        x = torch.cat([state, action], dim=-1)

        # Postprocess
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        return self.layer3(x)


class PolicyModel(nn.Module):
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_size: int = 256):
        super(PolicyModel, self).__init__()

        self.recurrent_layer = nn.LSTM(
            num_inputs, hidden_size, batch_first=True)

        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        for layer in [self.linear1, self.linear2]:
            layer.bias.data.fill_(B_INIT_VALUE)
            fanin_init(layer.weight)

        for layer in [self.mean_linear, self.log_std_linear]:
            layer.weight.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)
            layer.bias.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)

    def forward(self, state):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        _, (state, _) = self.recurrent_layer(state)

        state = state.squeeze(0)

        state = F.relu(self.linear1(state))
        state = F.relu(self.linear2(state))

        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        std = log_std.exp()

        std = torch.clamp(std, LOG_SIG_MIN, LOG_SIG_MAX)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std
