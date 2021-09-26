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

        self.query = nn.Linear(in_features=num_inputs + 1, out_features=16)

        self.key = nn.Linear(in_features=num_inputs + 1, out_features=16)

        self.value = nn.Linear(in_features=num_inputs + 1, out_features=16)

        for model_name in ['query', 'key', 'value']:
            setattr(
                self,
                model_name,
                nn.Sequential(
                    nn.Linear(in_features=num_inputs + 1, out_features=16),
                    nn.ReLU(),
                    nn.Linear(in_features=16, out_features=32),
                    nn.ReLU(),
                    nn.Linear(in_features=32, out_features=64),
                    nn.ReLU(),
                    nn.Linear(in_features=64, out_features=128),
                    nn.ReLU(),
                    nn.Linear(in_features=128, out_features=256),

                )
            )

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=128,
            batch_first=True
        )

        self.lstm = nn.RNN(
            input_size=256,
            hidden_size=256,
            batch_first=True
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        batch_size = state.shape[0]

        position_enconding = torch.arange(20).to('cuda:0')/20
        position_enconding = position_enconding.repeat(batch_size)
        position_enconding = position_enconding.reshape(batch_size, 20, 1)

        state = torch.cat([state, position_enconding], dim=-1)

        Q = self.query(state)
        K = self.key(state)
        V = self.value(state)

        context, _ = self.multi_head_attention(query=Q, key=K, value=V)

        # _, (context, _) = self.lstm(context)
        _, context = self.lstm(context)

        return context.squeeze(0)


class QModel(nn.Module):
    def __init__(
        self, hidden_size: int, post_attention: bool = False, **kwargs
    ):
        super(QModel, self).__init__()

        self.attention_base = AttentionBase(
            post_attention=post_attention, **kwargs)

        del kwargs['num_inputs']

        self.mlp_q = MLPQModel(
            num_inputs=256, hidden_sizes=512, **kwargs)

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
