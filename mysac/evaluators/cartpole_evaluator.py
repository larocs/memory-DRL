"""
This module defines a suit of tests to evaluate agents on the CartPoleEnv
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import torch
from gym import Env
from mysac.envs.pyrep_env import CartPoleEnv
from mysac.run_policy import env_from_specs, make_agent, policy_from_specs
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm

TEST_ACTUATION_SIGNAL_N_TIMES = 10


def test_actuation_signal(eval_folder: str, exp_path: str):
    """
    Tests the actuation signal based on the mass height

    Args:
        eval_folder: the folder where the evaluation data will be saved
        exp_path: the root folder to the experiment
    """
    specs['env']['headless'] = False

    env: CartPoleEnv = env_from_specs(specs=specs)
    policy = policy_from_specs(specs=specs, exp_path=exp_path)
    agent = make_agent(policy=policy, env=env)

    try:
        eval_folder = eval_folder + '/test_actuation_signal/'
        os.mkdir(eval_folder)
    except FileExistsError:
        print('Test actuation signal exsits, skipping...')

    for test in tqdm(range(TEST_ACTUATION_SIGNAL_N_TIMES),
                     desc='Testing actuation signal'):
        info = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=250,
            total_steps=250,
            deterministic=False
        )

        observations = info['observations']

        if env.buffer_for_rnn:
            # Get the 6th position of the last frame of every observation
            mass_z_positions = observations[:, -1, 6]

        else:
            mass_z_positions = observations[:, 6]

        mass_z_positions = mass_z_positions[1:]

        n_points = len(mass_z_positions)
        plt.title(f'Started from {mass_z_positions[0]}')
        plt.plot(range(n_points), mass_z_positions)
        plt.plot(range(n_points), n_points * [0.6])
        plt.savefig(eval_folder + f'/{test}')
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluates a trained policy '
                                     'in CartPoleEnv')

    parser.add_argument('--exp_path', type=str, required=True,
                        help='Source path for model binaries')

    args = parser.parse_args()

    experiment_folder = args.exp_path

    with open(experiment_folder + '/specs.json', 'r') as specs_file:
        specs = json.load(specs_file)

    eval_folder = experiment_folder + '/stats/eval/'
    os.mkdir(eval_folder)

    test_actuation_signal(eval_folder=eval_folder, exp_path=args.exp_path)
