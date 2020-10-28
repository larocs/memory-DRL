from typing import Tuple

import numpy as np
import torch
from gym import Env
from mysac.sac.utils import update_target_network
from torch import autograd
from torch.distributions import Normal


class SACAgent:
    """ Interface for a SACAgent """

    def __init__(self,
                 env: Env,

                 # Models
                 policy_model: torch.nn.Module,
                 q1_model: torch.nn.Module,
                 q1_target: torch.nn.Module,
                 q2_model: torch.nn.Module,
                 q2_target: torch.nn.Module,

                 # Hyperparams
                 gamma: float,
                 policy_lr: float,
                 q_lr: float,
                 alpha_lr: float,
                 tau: float,

                 # Metaparams
                 debug: bool = False):
        self.env = env

        # Models
        self.policy = policy_model

        self.q1 = q1_model
        self.q1_target = q1_target

        self.q2 = q2_model
        self.q2_target = q2_target

        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.target_entropy = -np.prod(self.env.action_space.shape).item()

        # Hyperparams
        self.gamma = gamma
        self.policy_lr = policy_lr
        self.q_lr = q_lr
        self.alpha_lr = alpha_lr
        self.tau = tau

        # Criterions
        self.q_criterion = torch.nn.MSELoss()
        self.v_criterion = torch.nn.MSELoss()

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=self.policy_lr
        )

        self.q1_optimizer = torch.optim.Adam(
            self.q1.parameters(),
            lr=self.q_lr
        )

        self.q2_optimizer = torch.optim.Adam(
            self.q2.parameters(),
            lr=self.q_lr
        )

        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha],
            lr=self.alpha_lr
        )

        # Metaparams
        self.debug = debug

    def get_action(
            self,
            observations: torch.tensor,
            reparametrize: bool = False,
            deterministic: bool = False,
            with_log_prob: bool = False,
            debug: bool = False) -> Tuple[torch.tensor, torch.tensor]:
        """ Returns an action for the given observation

        If the desired action is not deterministic, then we sample it from a
        Normal distribution using the reparametrization trick. The mean and std
        are obtained from the policy network for the given sample.

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
        mean, std = self.policy(observations)

        if deterministic:
            return torch.tanh(mean), mean, None

        if not reparametrize:
            sampled_action = Normal(mean, std).sample()

        else:
            # If the action is not deterministic, use the reparametrization
            # trick
            noise = Normal(
                torch.zeros(mean.size()),
                torch.ones(std.size()))\
                .sample()

            sampled_action = mean + std * noise

            sampled_action.requires_grad_()

        action = sampled_action
        tanh_action = torch.tanh(sampled_action)

        log_prob = None

        if with_log_prob:
            log_prob = Normal(mean, std).log_prob(action) - \
                torch.log(1 - tanh_action * tanh_action + 1e-6)

            log_prob = log_prob.sum(dim=1, keepdim=True)

        return tanh_action, action, log_prob

    def train_from_samples(self, batch):
        """ Makes a backward pass in a batch of transitions """
        rewards = batch['rewards']
        terminals = batch['terminals']
        observations = batch['observations']
        actions = batch['actions']
        next_observations = batch['next_observations']

        sampled_actions, _, log_prob = self.get_action(
            observations=observations,
            reparametrize=True,
            with_log_prob=True,
            debug=self.debug
        )

        # Alpha loss
        alpha_loss = -(self.log_alpha * (log_prob +
                                         self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp().detach()

        # Policy loss
        q_value = torch.min(
            self.q1(observations, sampled_actions),
            self.q2(observations, sampled_actions)
        )

        policy_loss = (alpha * log_prob - q_value).mean()

        # print('Alpha:', alpha)
        # print('Q value:', q_value)
        # print('Policy loss:', policy_loss)

        # Q lossess
        q1_prediction = self.q1(observations, actions)
        q2_prediction = self.q2(observations, actions)

        next_actions, _, next_log_prob = self.get_action(
            observations=next_observations,
            reparametrize=True,
            with_log_prob=True
        )

        # print('Next actions:', next_actions)
        # print('Next log prob:', next_log_prob)

        target_q_values = torch.min(
            self.q1_target(next_observations, next_actions),
            self.q2_target(next_observations, next_actions)
        ) - alpha * next_log_prob

        q_target = rewards + (1. - terminals) * self.gamma * target_q_values
        q1_loss = self.q_criterion(q1_prediction, q_target.detach())
        q2_loss = self.q_criterion(q2_prediction, q_target.detach())

        # Update all networks
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()

        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        update_target_network(
            q_model=self.q1, q_target=self.q1_target, tau=self.tau)
        update_target_network(
            q_model=self.q2, q_target=self.q2_target, tau=self.tau)
