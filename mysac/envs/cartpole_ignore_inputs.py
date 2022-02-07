from typing import List

import numpy as np
from gym import spaces
from mysac.envs import CARTPOLE_OBSERVATIONS
from mysac.envs.cartpole_perturb import CartPolePerturbationEnv
from mysac.envs.pyrep_env import CartPoleEnv


class CartPoleIgnoreStatesEnv(CartPoleEnv):
    """
    A Wrapper for CartPoleEnv that discards some of the output observations

    Args:
        ignore_obs: a list of observations to be discarded. Use the names
            defined in the `OBSERVATION` map
        args: list of arguments that will be passed to CartPoleEnv init
        kwargs: dict oof arguments that will be passed to CartPoleEnv init
    """

    def __init__(self, ignore_obs: List[str], *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.observations_to_ignore = [
            CARTPOLE_OBSERVATIONS[obs_name] for obs_name in ignore_obs
        ]

        num_obs = 10 - len(self.observations_to_ignore)

        act = np.array([2.]*2)
        self.action_space = spaces.Box(-act, act)

        obs = np.inf * np.ones((num_obs,))
        self.observation_space = spaces.Box(-obs, obs)

    def observe(self) -> np.array:
        """
        Returns the env observation at the current step, removing the
        observations specified in initialization
        """
        observation = super().observe()

        axis = 1 if self.buffer_for_rnn else 0

        return np.delete(observation, self.observations_to_ignore, axis=axis)
