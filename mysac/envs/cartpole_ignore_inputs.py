from gym import spaces
from typing import List

import numpy as np
from mysac.envs.pyrep_env import CartPoleEnv

OBSERVATIONS = {
    'cart_pos_x': 0, 'cart_pos_y': 1,
    'cart_vel_x': 2, 'cart_vel_y': 3,
    'mass_pos_x': 4, 'mass_pos_y': 5, 'mass_pos_z': 6,
    'mass_vel_x': 7, 'mass_vel_y': 8, 'mass_vel_z': 9
}


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
            OBSERVATIONS[obs_name] for obs_name in ignore_obs
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
