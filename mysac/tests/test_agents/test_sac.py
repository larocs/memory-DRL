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
        cls.agent_model = get_mocked_policy_model()

        cls.q1_model = get_mocked_q_model()
        cls.q1_target = get_mocked_q_model()

        cls.q2_model = get_mocked_q_model()
        cls.q2_target = get_mocked_q_model()

        cls.sac_agent = SACAgent(
            env=None,

            # Models
            agent_model=cls.agent_model,
            q1_model=cls.q1_model,
            q1_target=cls.q1_target,
            q2_model=cls.q2_model,
            q2_target=cls.q2_target,

            # Params
            gamma=0.99,
            agent_lr=0.1,
            q_lr=0.1,
            alpha_lr=0.1
        )

    @patch('mysac.sac.sac.Normal.sample')
    def test_get_actions(self, normal_mocked):
        """ Tests the interface for getting action """
        actions = torch.ones(1, DEFAULT_ACTION_SIZE)
        observations = torch.ones(1, DEFAULT_OBS_SIZE)

        expected_mean, expected_std = self.agent_model(observations)

        normal_mocked.side_effect = lambda x: x + 0.1

        with self.subTest('Test the deterministic output of the agent'):
            returned_tanh_mean, returned_mean = self.sac_agent.get_action(
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
            returned_tanh_mean, returned_mean = self.sac_agent.get_action(
                observations=observations, deterministic=False,
                reparametrize=False)

            # TODO: finish this test
