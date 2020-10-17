from typing import Tuple

import torch
from gym import Env
from torch.distributions import Normal


class SACAgent:
    """ Interface for a SACAgent """

    def __init__(self,
                 env: Env,

                 # Models
                 agent_model: torch.nn.Module,
                 q1_model: torch.nn.Module,
                 q1_target: torch.nn.Module,
                 q2_model: torch.nn.Module,
                 q2_target: torch.nn.Module,

                 # Hyperparams
                 gamma: float,
                 agent_lr: float,
                 q_lr: float,
                 alpha_lr: float):
        self.env = env

        # Models
        self.agent = agent_model

        self.q1 = q1_model
        self.q1_target = q1_target

        self.q2 = q2_model
        self.q2_target = q2_target

        # Hyperparams
        self.gamma = gamma
        self.agent_lr = agent_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr

    def get_action(
            self,
            observations: torch.tensor,
            reparametrize: bool = False,
            deterministic: bool = False) -> Tuple[torch.tensor, torch.tensor]:
        """ Returns an action for the given observation

        If the desired action is not deterministic, then we sample it from a
        Normal distribution using the reparametrization trick. The mean and std
        are obtained from the agent network for the given sample.

        Args:
            observations: the input observations
            reparametrize: if True, use the reparametrization trick. If not, it
                will assume the value won't be used for backpropr; simply
                uses the Normal distribution
            deterministic: if True, returns the best action (mean) for the
                given observation. Deterministic *never* uses reparametrization

        Returns:
            An action tensor ranging from 0 to 1 and a tensor representing the
            same action, but before the tanh activation
        """
        mean, std = self.agent(observations)

        if deterministic:
            return torch.tanh(mean), mean, None

        if not reparametrize:
            sampled_action = Normal(mean, std).sample()

        else:
            # If the action is not deterministic, use the reparametrization
            # trick
            sampled_action = (
                mean + std * Normal(
                    torch.zeros(mean.size()),
                    torch.zeros(std.size())
                ).sample()
            )

            sampled_action.requires_grad()

        action = sampled_action
        tanh_action = torch.tanh(sampled_action)

        log_prob = Normal(mean, std).log_prob(action) - \
            torch.log(1 - tanh_action * tanh_action + 1e-6)

        return tanh_action, action, log_prob
