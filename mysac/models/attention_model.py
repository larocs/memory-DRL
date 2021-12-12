# pylint: disable=no-member
import time

import torch
import torch.nn.functional as F
from matplotlib import pyplot
from mysac.models.mlp import PolicyModel as MLPPolicyModel
from mysac.models.mlp import QModel as MLPQModel
from numpy import intp, mod
from torch import nn
from torch.nn.modules.container import Sequential

# Numerical stability
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# Use the values as observed in Rlkit
B_INIT_VALUE = 0.1
W_INIT_VALUE = 3e-3

ENABLE_VIZ = False

if ENABLE_VIZ:
    pyplot.ion()


class AttentionBase(nn.Module):
    def __init__(
        self, num_inputs: int, num_outputs: int, pos_embedding: bool = False
    ):
        super(AttentionBase, self).__init__()

        self.pos_embedding = pos_embedding

        if pos_embedding:
            num_inputs += 1

        self.query = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=False
        )

        self.key = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=False
        )

        self.value = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs,
            bias=False
        )

        self.post_linear = nn.Linear(
            in_features=num_inputs,
            out_features=num_outputs
        )

        self.multi_head_attention = nn.MultiheadAttention(
            embed_dim=num_outputs,
            num_heads=1,
            batch_first=True
        )

        self.norm_1 = nn.LayerNorm(normalized_shape=num_outputs)
        self.norm_2 = nn.LayerNorm(normalized_shape=num_outputs)

    def forward(self, state: torch.tensor) -> torch.tensor:
        if self.pos_embedding:
            batch_size = state.shape[0]

            position_enconding = torch.arange(20).to('cuda:0')/20
            position_enconding = position_enconding.repeat(batch_size)
            position_enconding = position_enconding.reshape(batch_size, 20, 1)

            state = torch.cat([state, position_enconding], dim=-1)

        Q = self.query(state)
        K = self.key(state)
        V = self.value(state)

        context, attn_output_weights = self.multi_head_attention(
            query=Q,
            key=K,
            value=V,
            need_weights=ENABLE_VIZ
        )

        context = self.norm_1(context + state)

        linear_context = self.post_linear(context)

        if getattr(self, '_index_in_att_modules', None) == 10:
            pyplot.imshow(
                X=attn_output_weights.cpu().detach().numpy()[0],
                vmin=0,
                vmax=1
            )

            pyplot.pause(0.0001)

        return self.norm_2(context + linear_context)


class QModel(nn.Module):
    def __init__(
        self, hidden_size: int, post_attention: bool = False, **kwargs
    ):
        super(QModel, self).__init__()

        self.attention_base = nn.Sequential(
            AttentionBase(num_inputs=10, num_outputs=11, pos_embedding=True),
            AttentionBase(num_inputs=11, num_outputs=11),
            AttentionBase(num_inputs=11, num_outputs=11),
            AttentionBase(num_inputs=11, num_outputs=11),
            AttentionBase(num_inputs=11, num_outputs=11),
        )

        del kwargs['num_inputs']

        self.mlp_q = MLPQModel(num_inputs=11, hidden_sizes=32, **kwargs)

        print('Q Model:', self)

    def forward(self, state: torch.tensor, action: torch.tensor):
        state = self.attention_base.forward(state)

        state = state.mean(dim=1)

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

        self.attention_base = nn.Sequential(
            AttentionBase(num_inputs=10, num_outputs=11, pos_embedding=True),
            AttentionBase(num_inputs=11, num_outputs=11),
            AttentionBase(num_inputs=11, num_outputs=11),
            AttentionBase(num_inputs=11, num_outputs=11),
            AttentionBase(num_inputs=11, num_outputs=11),
        )

        self.mlp_policy = MLPPolicyModel(
            *args, num_inputs=11, hidden_sizes=32, **kwargs)

    def forward(self, state: torch.tensor):
        if not hasattr(self, '_indexed_modules') and ENABLE_VIZ:
            for module in self.attention_base.modules():
                if isinstance(module, Sequential):
                    for i, sub_module in enumerate(module):
                        sub_module._index_in_att_modules = i

            self._indexed_modules = True

        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        if False:
            pyplot.imshow(
                self.attention_base.forward(state).cpu().detach().numpy()[0]
            )

            pyplot.pause(0.0001)

        state = self.attention_base.forward(state).mean(dim=1)

        mean, std = self.mlp_policy.forward(observations=state)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std
