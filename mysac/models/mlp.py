from typing import Tuple

import torch
import torch.nn.functional as F
from mysac.models.utils import fanin_init
from torch import nn

# Use the values as observed in Rlkit
B_INIT_VALUE = 0.1
W_INIT_VALUE = 3e-3

# Default activation for every model
activation = F.relu

# Numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


class QModel(nn.Module):
    """ A deep model that receives observations and actions, and outputs a
    single real value.

    Args:
        obs_size: the observation dimension
        num_actions: the actions dimension
        hidden_sizes: the size for the hidden layers
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_sizes: int):
        super().__init__()

        self.layer1 = nn.Linear(num_inputs + num_actions, hidden_sizes)
        self.layer2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.layer3 = nn.Linear(hidden_sizes, hidden_sizes)
        self.layer4 = nn.Linear(hidden_sizes, 1)

        for layer in [self.layer1, self.layer2, self.layer4]:
            layer.bias.data.fill_(B_INIT_VALUE)
            fanin_init(layer.weight)

        self.layer4.weight.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)
        self.layer4.bias.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)

    def forward(self, observations, actions) -> torch.tensor:
        """ Returns a value for the pair observation/action """
        x = torch.cat([observations, actions], 1)

        x = activation(self.layer1(x))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))

        return self.layer4(x)


class PolicyModel(nn.Module):
    """ Deep model for computing agent actions given an observation

    Args:
        obs_size: the observation dimension
        num_actions: the actions dimension
        hidden_sizes: the size for the hidden layers
    """

    def __init__(self, num_inputs: int, num_actions: int, hidden_sizes: int):
        super().__init__()

        self.layer1 = nn.Linear(num_inputs, hidden_sizes)
        self.layer2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.layer3 = nn.Linear(hidden_sizes, hidden_sizes)

        self.mean = nn.Linear(hidden_sizes, num_actions)
        self.log_std = nn.Linear(hidden_sizes, num_actions)

        for layer in [self.layer1, self.layer2, self.layer3]:
            layer.bias.data.fill_(B_INIT_VALUE)
            fanin_init(layer.weight)

        for layer in [self.mean, self.log_std]:
            layer.weight.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)
            layer.bias.data.uniform_(-W_INIT_VALUE, W_INIT_VALUE)

    def forward(self, observations) -> Tuple[torch.tensor, torch.tensor]:
        """ Returns an mean and a std for the actions Normal distribution given
        observation """
        x = activation(self.layer1(observations))
        x = activation(self.layer2(x))
        x = activation(self.layer3(x))

        mean = self.mean(x)
        log_std = self.log_std(x)

        std = log_std.exp()

        return mean, torch.clamp(std, LOG_SIG_MIN, LOG_SIG_MAX)
