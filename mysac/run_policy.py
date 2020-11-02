# pylint: disable=no-member
# pylint: disable=not-callable

import argparse
import json

import torch

from mysac.envs.pyrep_env import CartPoleEnv
from mysac.sac.sac import SACAgent

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

    policy = PolicyModel(**specs['models']['policy'])
    policy.load_state_dict(torch.load(args.exp_path + '/models/policy.pt'))
    policy.eval()

    if specs['env']['name'] == 'CartPole':
        env_specs = specs['env']['specs']
        env_specs['buffer_for_rnn'] = True
        env_specs['headless'] = False

        env = CartPoleEnv(**env_specs)

    obs = env.reset()
    while True:
        action, _ = policy(torch.tensor(obs))

        obs, _, _, _ = env.step(action)
