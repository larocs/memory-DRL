from mysac.models.mlp import QModel, PolicyModel

import torch
from typing import Tuple


def stack_frames(observations: torch.tensor, num_inputs: int) -> torch.tensor:
    if len(observations.shape) == 3:
        batch_size = observations.shape[0]

        reshape_args = (batch_size, num_inputs)

    else:
        reshape_args = (num_inputs,)

    return observations.reshape(*reshape_args)


class StackedQModel(QModel):
    """ A deep model that receives observations and actions, and outputs a
    single real value.

    Args:
        obs_size: the observation dimension
        num_actions: the actions dimension
        hidden_sizes: the size for the hidden layers
    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_sizes: int,
        num_frames: int
    ):
        super().__init__(
            num_inputs=num_inputs * num_frames,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes
        )

        self.num_inputs = num_inputs * num_frames

    def forward(self, observations, actions) -> torch.tensor:
        """ Returns a value for the pair observation/action """
        observations = stack_frames(
            observations=observations,
            num_inputs=self.num_inputs
        )

        return super().forward(observations=observations, actions=actions)


class StackedPolicyModel(PolicyModel):
    """ Deep model for computing agent actions given an observation

    Args:
        obs_size: the observation dimension
        num_actions: the actions dimension
        hidden_sizes: the size for the hidden layers
    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_sizes: int,
        num_frames: int
    ):
        super().__init__(
            num_inputs=num_inputs * num_frames,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes
        )

        self.num_inputs = num_inputs * num_frames

    def forward(
        self,
        observations: torch.tensor
    ) -> Tuple[torch.tensor, torch.tensor]:
        """ Returns an mean and a std for the actions Normal distribution given
        observation """
        observations = stack_frames(
            observations=observations,
            num_inputs=self.num_inputs
        )

        return super().forward(observations=observations)
