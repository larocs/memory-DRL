import numpy as np
from typing import Tuple, List


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

        print('Observations:', self.observations)
        self.observations = self._roll_and_add(self.observations, observations)
        self.next_observations = self._roll_and_add(
            self.next_observations, next_observations)
        self.rewards = self._roll_and_add(self.rewards, rewards)
        self.actions = self._roll_and_add(self.actions, actions)
        self.terminals = self._roll_and_add(self.terminals, terminals)

        if self.ix < self.size:
            self.ix += new_steps

    def sample(self, n_samples: int) -> Tuple[List[float], List[float],
                                              List[float], List[float],
                                              List[float]]:
        """ Sample n transitions from buffer """
        sampled_indexes = np.random.random_integers(0, self.ix, n_samples)

        observations = self.observations[sampled_indexes]
        next_observations = self.next_observations[sampled_indexes]
        rewards = self.rewards[sampled_indexes]
        actions = self.actions[sampled_indexes]
        terminals = self.terminals[sampled_indexes]

        return (
            observations,
            next_observations,
            rewards,
            actions,
            terminals
        )
