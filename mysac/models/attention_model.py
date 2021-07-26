#pylint: disable=no-member
import time

import torch
import torch.nn.functional as F
from mysac.models.mlp import PolicyModel as MLPPolicyModel
from mysac.models.mlp import QModel as MLPQModel
from numpy import intp
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
                nn.Linear(num_inputs, 2 * num_inputs),
                nn.ReLU(),
                nn.Linear(2 * num_inputs, 2 * num_inputs),
                nn.ReLU(),
                nn.Linear(2 * num_inputs, num_inputs),
                nn.Softmax(dim=1)
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
            state = state * attention_mask

            state = self._forward_lstm(state)

        return state


class QModel(nn.Module):
    def __init__(
        self, hidden_size: int, post_attention: bool = False, **kwargs
    ):
        super(QModel, self).__init__()

        self.attention_base = AttentionBase(
            post_attention=post_attention, **kwargs)

        del kwargs['num_inputs']

        self.mlp_q = MLPQModel(
            num_inputs=256, hidden_sizes=hidden_size, **kwargs)

        print('Q Model:', self)

    def forward(self, state: torch.tensor, action: torch.tensor):
        state = self.attention_base.forward(state=state)

        return self.mlp_q.forward(observations=state, actions=action)


class PolicyModel(nn.Module):
    def __init__(
        self,
        *args,
        hidden_size: int,
        num_inputs: int,
        post_attention: bool = False,
        **kwargs
    ):
        super(PolicyModel, self).__init__()

        self.attention_base = AttentionBase(
            *args, post_attention=post_attention, num_inputs=num_inputs,
            **kwargs)

        self.mlp_policy = MLPPolicyModel(
            *args, hidden_sizes=512, num_inputs=256, **kwargs)

    def forward(self, state: torch.tensor):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        state = self.attention_base.forward(state=state)

        mean, std = self.mlp_policy.forward(observations=state)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std
