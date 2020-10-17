import torch
import torch.nn.functional as F
from torch import nn


class QModel(nn.Module):
    def __init__(self, obs_size: int, num_actions: int, hidden_sizes: int):
        super(QModel).__init__()

        self.layer1 = nn.Linear(obs_size + num_actions, hidden_sizes)
        self.layer2 = nn.Linear(hidden_sizes, hidden_sizes)
        self.layer3 = nn.Linear(hidden_sizes, 1)

    def forward(self, observations, actions):
        x = torch.cat([observations, actions], 1)
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = self.layer3(x)


class AgentModel(nn.Module):
    def __init__(self):
        pass
