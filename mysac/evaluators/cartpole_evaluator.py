"""
This module defines a suit of tests to evaluate agents on the CartPoleEnv
"""
import argparse
import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from mysac.envs.cartpole_perturb import CartPolePerturbationEnv
from mysac.envs.pyrep_env import CartPoleEnv
from mysac.run_policy import env_from_specs, make_agent, policy_from_specs
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm

REPEAT_TEST_N_TIMES = 10


def build_everything_from_specs(specs,
                                exp_path: str,
                                env_class: CartPoleEnv = CartPoleEnv,
                                headless: bool = True) \
        -> Tuple[CartPoleEnv, torch.nn.Module, SACAgent]:
    """
    Build the env and policy from a specs dict

    Args:
        specs: the specs dict, as saved by the trainer
        exp_path: the path to the experiment, where the policy model will be
            loaded from
        headless: if True, forces the Coppelia sim to be rendered
        env_class: a subclass of CartPoleEnv that will be used to create the
            env with the given specs
    """
    env: CartPoleEnv = env_from_specs(
        specs=specs,
        headless=headless,
        env_class=env_class,
    )

    policy = policy_from_specs(specs=specs, exp_path=exp_path)
    agent = make_agent(policy=policy, env=env)

    return env, policy, agent


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

    for test in tqdm(range(REPEAT_TEST_N_TIMES),
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


def test_perturbation(specs, eval_folder: str, exp_path: str):
    """
    Applies perturbations to the pole mass everytime it gets stable (see
    `CartPolePerturbationEnv` for details)

    Args:
        specs: the specs dict in the experiment folder
        eval_folder: the folder where the evaluation results will be saved
        exp_path: the path to the experiment folder
    """
    STEPS = 750

    env, _, agent = build_everything_from_specs(
        specs,
        env_class=CartPolePerturbationEnv,
        headless=False,
        exp_path=exp_path
    )

    for _ in tqdm(range(REPEAT_TEST_N_TIMES), desc='Test '
                  'perturbation'):
        BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=STEPS,
            total_steps=STEPS,
            deterministic=True
        )

    env.save_eval(eval_folder=eval_folder)


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
    test_perturbation(specs=specs, eval_folder=eval_folder,
                      exp_path=args.exp_path)
