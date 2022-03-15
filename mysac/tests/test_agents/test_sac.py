# pylint: disable=no-member
# pylint: disable=not-callable
from unittest import TestCase
from unittest.mock import MagicMock, patch

import torch
from mysac.sac.sac import SACAgent
from mysac.tests.mock.mock_models import (DEFAULT_ACTION_SIZE,
                                          DEFAULT_OBS_SIZE,
                                          get_mocked_policy_model,
                                          get_mocked_q_model)


class SACAgentTest(TestCase):
    """ Tests the forward and backward for the SACAgent """

    @classmethod
    def setUpClass(cls):
        cls.policy_model = get_mocked_policy_model()

        cls.q1_model = get_mocked_q_model()
        cls.q1_target = get_mocked_q_model()

        cls.q2_model = get_mocked_q_model()
        cls.q2_target = get_mocked_q_model()

        cls.env = MagicMock()
        cls.env.action_space.shape = (-DEFAULT_ACTION_SIZE,
                                      DEFAULT_ACTION_SIZE)

    def get_sac_agent(self):
        return SACAgent(
            env=self.env,

            # Models
            policy_model=self.policy_model,
            q1_model=self.q1_model,
            q1_target=self.q1_target,
            q2_model=self.q2_model,
            q2_target=self.q2_target,

            # Params
            gamma=0.99,
            policy_lr=0.1,
            q_lr=0.1,
            alpha_lr=0.1,
            tau=0.1
        )

    @patch('mysac.sac.sac.Normal.sample')
    def test_get_actions(self, normal_mocked):
        """ Tests the interface for getting action """
        sac_agent = self.get_sac_agent()

        observations = torch.ones(1, DEFAULT_OBS_SIZE)

        expected_mean, expected_std = self.policy_model(observations)

        normal_mocked.return_value = expected_mean + 0.1

        with self.subTest('Test the deterministic output of the policy'):
            returned_tanh_mean, returned_mean, _ = sac_agent.get_action(
                observations=observations, deterministic=True)

            self.assertEqual(
                torch.tanh(expected_mean).tolist(),
                returned_tanh_mean.tolist()
            )

            self.assertEqual(
                expected_mean.tolist(),
                returned_mean.tolist()
            )

        with self.subTest('Test the action sampling without reparam trick'):
            returned_tanh_mean, returned_mean, _ = sac_agent.get_action(
                observations=observations, deterministic=False,
                reparametrize=False)

            # TODO: finish this test

    @patch('mysac.sac.sac.Normal.log_prob')
    @patch('mysac.sac.sac.Normal.sample')
    def test_train_from_batch(self, sample_mock: MagicMock,
                              log_prob: MagicMock):
        """
        Tests the backward pass for the SACAgent
        """

        sample_mock.return_value = torch.ones(1, 3)
        log_prob.return_value = torch.ones(1, 3)

        # Mock a single transition:
        #  R: 10
        #  D: 0
        #  S: (1, 1)
        #  A: (2, 2, 2)
        #  S': (2, 2)
        batch = dict()
        batch['rewards'] = torch.tensor([[10]], dtype=torch.float)
        batch['terminals'] = torch.tensor([[0]], dtype=torch.float)
        batch['observations'] = torch.tensor([[1, 1]], dtype=torch.float)
        batch['next_observations'] = torch.tensor([[2, 2]], dtype=torch.float)
        batch['actions'] = torch.tensor(
            [[-0.5691, -0.1300,  0.2416]], dtype=torch.float)

        sac_agent = self.get_sac_agent()
        sac_agent.debug = True

        sac_agent.train_from_samples(batch)

        self.assertEqual(
            torch.tensor(
                [p.sum() for p in sac_agent.policy.parameters()]
            ).sum().item(),
            -2.8618392944335938,
            'Test the backward for the policy model'
        )

        self.assertEqual(
            torch.tensor(
                [p.sum() for p in sac_agent.q1.parameters()]
            ).sum().item(),
            0.3352656662464142,
            'Test the backward for the q1 model'
        )

        self.assertEqual(
            torch.tensor(
                [p.sum() for p in sac_agent.q1_target.parameters()]
            ).sum().item(),
            0.15526564419269562,
            'Test the backward for the q1_target model'
        )

        self.assertEqual(
            torch.tensor(
                [p.sum() for p in sac_agent.q2.parameters()]
            ).sum().item(),
            0.3352656662464142,
            'Test the backward for the q2 model'
        )

        self.assertEqual(
            torch.tensor(
                [p.sum() for p in sac_agent.q2_target.parameters()]
            ).sum().item(),
            0.15526564419269562,
            'Test the backward for the q2_target model'
        )
