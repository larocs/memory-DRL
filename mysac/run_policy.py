# pylint: disable=no-member
# pylint: disable=not-callable

import argparse
import json

import torch

from mysac.envs.pyrep_env import CartPoleEnv
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained policy from '
                                     'specs')

    parser.add_argument('--exp_path', type=str,
                        help='Source path for model binaries')

    args = parser.parse_args()

    experiment_folder = args.exp_path

    with open(experiment_folder + '/specs.json', 'r') as specs_file:
        specs = json.load(specs_file)

    print(specs)
    # Select the model
    if specs['models']['mode'] == 'rnn':
        from mysac.models.rnn_models import PolicyModel, QModel

    else:
        from mysac.models.mlp import PolicyModel, QModel

    policy = torch.load(args.exp_path + '/models/policy.pt')
    policy.eval()

    if specs['env']['name'] == 'CartPole':
        env_specs = specs['env']['specs']

        buffer_for_rnn = env_specs.get('buffer_for_rnn', None)
        if buffer_for_rnn is None:
            env_specs['buffer_for_rnn'] = False

        env_specs['headless'] = False

        env = CartPoleEnv(**env_specs)

    agent = SACAgent(
        env=env,
        policy_model=policy,
        q1_model=policy,
        q1_target=policy,
        q2_model=policy,
        q2_target=policy,

        # Hyperparams
        gamma=0,
        policy_lr=0,
        q_lr=0,
        alpha_lr=0,
        tau=0
    )

    BasicTrajectorySampler.sample_trajectory(
        env=env,
        agent=agent,
        max_steps_per_episode=250,
        total_steps=1e6,
        deterministic=True
    )
