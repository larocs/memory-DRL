# pylint: disable=no-member
# pylint: disable=not-callable
from typing import List, Tuple

import numpy as np
import torch
from mysac.utils import get_device


class NumpySampledBuffer:
    """ Buffer whose sample is implemented with Numpy

    Args:
        size: the max size of the buffer
        observation_size: the size of the observation
        action_size: the size of the action
    """

    def __init__(self, size: int, observation_size: int, action_size: int):
        self.observations = np.zeros((size, observation_size))
        self.next_observations = np.zeros((size, observation_size))
        self.rewards = np.zeros((size, 1))
        self.actions = np.zeros((size, action_size))
        self.terminals = np.zeros((size, 1))

        self.ix = 0
        self.size = size

        self.device = get_device()

    @staticmethod
    def _roll_and_add(array: np.array, new_steps: np.array):
        n_steps = new_steps.shape[0]

        array = np.roll(array, n_steps, axis=0)

        array[:n_steps] = new_steps

        return array

    def add_trajectory(
        self,
        observations: np.array,
        next_observations: np.array,
        rewards: np.array,
        actions: np.array,
        terminals: np.array
    ):
        """ Add a trajectory """
        new_steps = observations.shape[0]

        # Sanity check
        assert next_observations.shape[0] == new_steps
        assert rewards.shape[0] == new_steps
        assert actions.shape[0] == new_steps
        assert terminals.shape[0] == new_steps

        self.observations = self._roll_and_add(self.observations, observations)
        self.next_observations = self._roll_and_add(
            self.next_observations, next_observations)
        self.rewards = self._roll_and_add(self.rewards, rewards)
        self.actions = self._roll_and_add(self.actions, actions)
        self.terminals = self._roll_and_add(self.terminals, terminals)

        self.ix = min(self.ix + new_steps, self.size - 1)

    def sample(self, n_samples: int) -> Tuple[List[float], List[float],
                                              List[float], List[float],
                                              List[float]]:
        """ Sample n transitions from buffer """
        sampled_indexes = np.random.random_integers(0, self.ix - 1, n_samples)

        observations = torch.tensor(
            self.observations[sampled_indexes],
            dtype=torch.float,
            device=self.device
        )

        next_observations = torch.tensor(
            self.next_observations[sampled_indexes],
            dtype=torch.float,
            device=self.device
        )

        rewards = torch.tensor(
            self.rewards[sampled_indexes],
            dtype=torch.float,
            device=self.device
        )

        actions = torch.tensor(
            self.actions[sampled_indexes],
            dtype=torch.float,
            device=self.device
        )

        terminals = torch.tensor(
            self.terminals[sampled_indexes],
            dtype=torch.float,
            device=self.device
        )

        return {
            'observations': observations,
            'next_observations': next_observations,
            'rewards': rewards,
            'actions': actions,
            'terminals': terminals
        }


class NumpySampledBufferForRNN(NumpySampledBuffer):
    """
    Buffer whose sample is implemented with Numpy

    Args:
        size: the max size of the buffer
        observation_size: the size of the observation
        action_size: the size of the action
    """

    def __init__(self, size: int, observation_size: int, action_size: int,
                 frames: int):
        self.observations = np.zeros((size, frames, observation_size))
        self.next_observations = np.zeros((size, frames, observation_size))
        self.rewards = np.zeros((size, 1))
        self.actions = np.zeros((size, action_size))
        self.terminals = np.zeros((size, 1))

        self.ix = 0
        self.size = size

        self.device = get_device()
