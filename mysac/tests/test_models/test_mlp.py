from unittest import TestCase

import torch
from mysac.models.mlp import PolicyModel, QModel

DEFAULT_OBS_SIZE = 2
DEFAULT_ACTION_SIZE = 3
DEFAULT_HIDDEN_SIZE = 5


class QModelTest(TestCase):
    def test_forward_interface(self):
        """ Tests the foward interface for MLP Q Function """
        q = QModel(
            num_inputs=DEFAULT_OBS_SIZE,
            num_actions=DEFAULT_ACTION_SIZE,
            hidden_sizes=DEFAULT_HIDDEN_SIZE
        )

        batch_sizes = [1, 2, 3]

        for batch_size in batch_sizes:
            with self.subTest('Tests that the model outputs the correct shape '
                              f'for a batch of size {batch_size}'):
                # Tests a single batch of observations/actions
                observations = torch.ones((batch_size, DEFAULT_OBS_SIZE))
                actions = torch.ones((batch_size, DEFAULT_ACTION_SIZE))

                output = q(observations, actions)

                self.assertEqual(output.shape, (batch_size, 1))


class PolicyModelTest(TestCase):
    def test_forward_interface(self):
        """ Tests the foward interface for MLP Policy Function """
        pi = PolicyModel(
            num_inputs=DEFAULT_OBS_SIZE,
            num_actions=DEFAULT_ACTION_SIZE,
            hidden_sizes=DEFAULT_HIDDEN_SIZE
        )

        batch_sizes = [1, 2, 3]

        for batch_size in batch_sizes:
            with self.subTest('Tests that the model outputs the correct shape '
                              f'for a batch of size {batch_size}'):
                # Tests a single batch of observations/actions
                observations = torch.ones((batch_size, DEFAULT_OBS_SIZE))

                mean, log_std = pi(observations)

                self.assertEqual(mean.shape, (batch_size, DEFAULT_ACTION_SIZE))
                self.assertEqual(
                    log_std.shape, (batch_size, DEFAULT_ACTION_SIZE))
