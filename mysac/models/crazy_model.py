import torch
from mysac.models.attention_model import AttentionBase
from mysac.models.mlp import PolicyModel as MLPPolicyModel
from mysac.models.mlp import QModel as MLPQModel
from torch import nn


class CrazyAttentionLayer(nn.Module):
    def __init__(self):
        super(CrazyAttentionLayer, self).__init__()

        self.cartpos_att = nn.Sequential(
            AttentionBase(
                num_inputs=2,
                num_outputs=3,
                pos_embedding=True
            ),
            AttentionBase(
                num_inputs=3,
                num_outputs=3
            )
        )

        self.cartvel_att = nn.Sequential(
            AttentionBase(
                num_inputs=2,
                num_outputs=3,
                pos_embedding=True
            ),
            AttentionBase(
                num_inputs=3,
                num_outputs=3
            )
        )

        self.masspos_att = nn.Sequential(
            AttentionBase(
                num_inputs=3,
                num_outputs=4,
                pos_embedding=True
            ),
            AttentionBase(
                num_inputs=4,
                num_outputs=4
            )
        )

        self.massvel_att = nn.Sequential(
            AttentionBase(
                num_inputs=3,
                num_outputs=4,
                pos_embedding=True
            ),
            AttentionBase(
                num_inputs=4,
                num_outputs=4
            )
        )

        self.cat_att = nn.Sequential(
            AttentionBase(
                num_inputs=14,
                num_outputs=15,
                pos_embedding=True
            ),
            AttentionBase(
                num_inputs=15,
                num_outputs=15
            )
        )

    def forward(self, state: torch.tensor) -> torch.tensor:
        """
        """
        cartpos = state[:, :, 0:2]
        cartvel = state[:, :, 2:4]
        masspos = state[:, :, 4:7]
        massvel = state[:, :, 7:]

        cartpos = self.cartpos_att(cartpos)
        cartvel = self.cartvel_att(cartvel)
        masspos = self.masspos_att(masspos)
        massvel = self.massvel_att(massvel)

        return self.cat_att(
            torch.cat(
                tensors=(cartpos, cartvel, masspos, massvel),
                dim=-1
            )
        )


class QModel(nn.Module):
    def __init__(
        self, hidden_size: int, post_attention: bool = False, **kwargs
    ):
        super(QModel, self).__init__()

        self.attention_base = CrazyAttentionLayer()

        del kwargs['num_inputs']

        self.mlp_q = MLPQModel(num_inputs=15, hidden_sizes=32, **kwargs)

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

        self.attention_base = CrazyAttentionLayer()

        self.mlp_policy = MLPPolicyModel(
            *args, num_inputs=15, hidden_sizes=32, **kwargs)

    def forward(self, state: torch.tensor):
        if len(state.shape) == 2:
            state = state.unsqueeze(0)

        state = self.attention_base.forward(state).mean(dim=1)

        mean, std = self.mlp_policy.forward(observations=state)

        if mean.shape[0] == 1:
            mean = mean.squeeze(0)
            std = std.squeeze(0)

        return mean, std
