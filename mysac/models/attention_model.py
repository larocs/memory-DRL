#pylint: disable=no-member
import torch
import torch.nn.functional as F
from torch import nn

# Numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Use the values as observed in Rlkit
B_INIT_VALUE = 0.1
W_INIT_VALUE = 3e-3


class AttentionBase(nn.Module):
    def __init__(
        self, num_inputs: int, num_actions: int, hidden_size: int = 256
    ):
        super(AttentionBase, self).__init__()

        self.recurrent_layer = nn.LSTM(
            num_inputs, hidden_size, batch_first=True)

        self.attention_1 = nn.Linear(hidden_size, hidden_size)
        self.attention_2 = nn.Linear(hidden_size, hidden_size)

        self.softmax = torch.nn.Softmax(dim=1)

        self.flatten_1 = torch.nn.Linear(20 * hidden_size, 512)
        self.flatten_2 = torch.nn.Linear(512, 256)

    def forward(self, state: torch.tensor):
        state, *_ = self.recurrent_layer(state)

        attention_mask = F.relu(self.attention_1(state))
        attention_mask = self.softmax(self.attention_2(attention_mask))

        state = state * attention_mask
        state = torch.flatten(state, -2, -1)

        state = F.relu(self.flatten_1(state))
        state = F.relu(self.flatten_2(state))

        return state


class QModel(AttentionBase):
    def __init__(self, num_actions: int, **kwargs):
        super().__init__(num_actions=num_actions, **kwargs)

        self.merge = nn.Linear(256 + num_actions, 1)

    def forward(self, state: torch.tensor, action: torch.tensor):
        state = super().forward(state=state)

        return self.merge(torch.cat([state, action], dim=-1))


class PolicyModel(AttentionBase):
    def __init__(self,
                 num_inputs: int,
                 num_actions: int,
                 hidden_size: int = 256):
        super().__init__(
            num_inputs=num_inputs,
            num_actions=num_actions,
            hidden_size=hidden_size
        )

        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)

    def forward(self, state: torch.tensor):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        state = super().forward(state=state)

        mean = self.mean_linear(state)
        log_std = self.log_std_linear(state)
        std = log_std.exp()

        std = torch.clamp(std, LOG_SIG_MIN, LOG_SIG_MAX)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std
