from hashlib import new

import torch
from mysac.models.attention_model import AttentionBase
from mysac.models.mlp import LOG_SIG_MAX, LOG_SIG_MIN
from mysac.models.mlp import PolicyModel as MLPPolicyModel
from mysac.models.mlp import QModel as MLPQModel
from torch import nn


class CrazyAttentionLayer(nn.Module):
    def __init__(self):
        super(CrazyAttentionLayer, self).__init__()

        self.intraframe_att = AttentionBase(
            num_inputs=1,
            num_outputs=2,
            pos_embedding=True,
            num_heads=2,
            num_frames=10
        )

        self.interframe_att = AttentionBase(
            num_inputs=20,
            num_outputs=21,
            pos_embedding=True,
            num_frames=10
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        """
        """
        new_batch = []

        for batch in state:
            frames = batch.reshape(10, 10).unsqueeze(-1)

            new_frames = self.intraframe_att(frames)
            new_frames = new_frames.reshape(10, 20).unsqueeze(0)

            new_batch.append(new_frames)

        new_batch = torch.cat(tensors=new_batch)

        return self.interframe_att(new_batch)


class QModel(nn.Module):
    def __init__(
        self, hidden_size: int, post_attention: bool = False, **kwargs
    ):
        super(QModel, self).__init__()

        self.attention_base = CrazyAttentionLayer()

        del kwargs['num_inputs']

        self.linear = nn.Sequential(
            nn.Linear(
                in_features=212,
                out_features=128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=2
            )
        )

        # print('Q Model:', self)

    def forward(self, state: torch.tensor, action: torch.tensor):
        state = self.attention_base.forward(state)
        state = state.reshape(-1, 210)
        state = torch.cat([state, action], 1)

        return self.linear(state)


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

        self.attention_base = CrazyAttentionLayer()

        self.mean_linear = nn.Sequential(
            nn.Linear(
                in_features=210,
                out_features=128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=2
            )
        )

        self.std_linear = nn.Sequential(
            nn.Linear(
                in_features=210,
                out_features=128
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=128,
                out_features=64
            ),
            nn.ReLU(),
            nn.Linear(
                in_features=64,
                out_features=2
            )
        )

    def forward(self, state: torch.tensor):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        state = self.attention_base.forward(state).reshape(-1, 210)

        mean = self.mean_linear(state)
        std = self.std_linear(state).exp()

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, torch.clamp(std, LOG_SIG_MIN, LOG_SIG_MAX)
