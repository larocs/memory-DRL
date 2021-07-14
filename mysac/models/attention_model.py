#pylint: disable=no-member
import time

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
        self, num_inputs: int, num_actions: int, hidden_size: int = 256,
        post_attention: bool = True
    ):
        super(AttentionBase, self).__init__()
        print('Attention model; post_attention:', post_attention)

        self.post_attention = post_attention

        self.recurrent_layer = nn.LSTM(
            num_inputs, hidden_size, batch_first=True)

        if self.post_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.Softmax(dim=1)
            )

        else:
            self.attention = nn.Sequential(
                nn.Linear(num_inputs, num_inputs),
                nn.Softmax(dim=1)
            )

            self.dense = nn.Sequential(
                nn.Linear(num_inputs, num_inputs),
                nn.ReLU(),
                nn.Linear(num_inputs, num_inputs),
            )

    def _forward_lstm(self, state: torch.tensor) -> torch.tensor:
        _, (state, _) = self.recurrent_layer(state)

        return state.squeeze(0)

    def forward(self, state: torch.tensor) -> torch.tensor:
        if self.post_attention:
            state = self._forward_lstm(state)
            attention_mask = self.attention(state)

            state = state * attention_mask

        else:
            attention_mask = self.attention(state)
            state = self.dense(state)

            state = self._forward_lstm(state)

        return state


class QModel(AttentionBase):
    def __init__(self, num_actions: int, hidden_size: int, **kwargs):
        super().__init__(
            num_actions=num_actions,
            hidden_size=hidden_size,
            **kwargs
        )

        self.merge = nn.Linear(hidden_size + num_actions, 1)

    def forward(self, state: torch.tensor, action: torch.tensor):
        state = super().forward(state=state)

        return self.merge(torch.cat([state, action], dim=-1))


class PolicyModel(AttentionBase):
    def __init__(
            self,
            num_inputs: int,
            num_actions: int,
            hidden_size: int = 256,
            **kwargs
    ):

        super().__init__(
            num_inputs=num_inputs,
            num_actions=num_actions,
            hidden_size=hidden_size,
            **kwargs
        )

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)

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
