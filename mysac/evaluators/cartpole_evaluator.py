"""
This module defines a suit of tests to evaluate agents on the CartPoleEnv
"""
import argparse
import json
import os
from sys import path
from typing import Callable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from mysac.envs.cartpole_perturb import CartPolePerturbationEnv
from mysac.envs.pyrep_env import CartPoleEnv
from mysac.run_policy import env_from_specs, make_agent, policy_from_specs
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm

REPEAT_TEST_N_TIMES = 50


def reset_random_seed():
    """
    Resets the RNG for Numpy and Torch
    """
    torch.manual_seed(0)
    np.random.seed(0)


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


def basic_run(
        env: CartPoleEnv, agent: SACAgent, deterministic: bool = False
) -> np.array:
    """
    Sample a trajectory from the given env/agent

    Args:
        env: the environment where the steps will be sampled from
        agent: the agent for sampling the actions given the observations
        deterministic: if True, use the best actions only

    Returns:
        A NumPy arrat containing all the z positions for the pole
    """
    info = BasicTrajectorySampler.sample_trajectory(
        env=env,
        agent=agent,
        max_steps_per_episode=250,
        total_steps=250,
        deterministic=deterministic
    )

    observations = info['observations']

    if env.buffer_for_rnn:
        # Get the 6th position of the last frame of every observation
        mass_z_positions = observations[:, -1, 6]

    else:
        mass_z_positions = observations[:, 6]

    mass_z_positions = mass_z_positions[1:]

    return mass_z_positions


def test_num_stable_frames(
        specs, eval_folder: str, exp_path: str):
    """
    Measure mean number of frames where the mass height was above 0.55m

    Args:
        eval_folder: the folder where the evaluation data will be saved
        exp_path: the root folder to the experiment
    """
    POMDP_MODES = [
        None,
        'noise',
        'zero(cart_pos_x,cart_pos_y)',
        'zero(cart_vel_x,cart_vel_y)',
        'zero(mass_pos_x,mass_pos_y,mass_pos_z)',
        'zero(mass_vel_x,mass_vel_y,mass_vel_z)'
    ]

    specs['env']['headless'] = False

    try:
        eval_folder = eval_folder + '/test_num_stable_frames/'
        os.mkdir(eval_folder)

    except FileExistsError:
        print('Test num stable frames exists, skipping...')
        return

    for pomdp_mode in POMDP_MODES:
        print('Testing', pomdp_mode)

        sub_eval_folder = eval_folder + str(pomdp_mode) + '/'
        os.mkdir(sub_eval_folder)

        specs['env']['specs']['pomdp_mode'] = pomdp_mode

        env: CartPoleEnv = env_from_specs(specs=specs)
        policy = policy_from_specs(specs=specs, exp_path=exp_path)
        agent = make_agent(policy=policy, env=env)

        df = pd.DataFrame(
            columns=['test', 'z']
        )

        for test in tqdm(range(REPEAT_TEST_N_TIMES), desc=pomdp_mode):
            mass_z_positions = basic_run(
                env=env,
                agent=agent,
                deterministic=True
            )

            n_points = len(mass_z_positions)

            df = df.append(
                pd.DataFrame(
                    {
                        'test': n_points * [test],
                        'z': mass_z_positions
                    }
                )
            )

        env.pr.shutdown()

        df.to_csv(
            path_or_buf=sub_eval_folder + 'z_history.csv',
            mode='a'
        )

        df['success'] = df['z'] > 0.55
        mean = df.groupby('test').sum()['success'].mean()

        os.mkdir(sub_eval_folder + str(mean))


def test_actuation_signal(
        eval_folder: str, exp_path: str,
        increase_difficulty_callback: Callable[[CartPoleEnv], Optional[bool]]):
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
        callback_name = increase_difficulty_callback.name
        eval_folder = eval_folder + f'/test_actuation_signal_{callback_name}/'
        os.mkdir(eval_folder)

    except FileExistsError:
        print('Test actuation signal exsits, skipping...')
        return

    param_value = True
    while param_value is not None:
        param_value = increase_difficulty_callback(env=env)
        os.mkdir(eval_folder + f'/{param_value}')
        reset_random_seed()

        for test in range(REPEAT_TEST_N_TIMES):
            mass_z_positions = basic_run(
                env=env,
                agent=agent,
                deterministic=False
            )

            n_points = len(mass_z_positions)
            plt.title(f'Started from {mass_z_positions[0]}')
            plt.plot(range(n_points), mass_z_positions)
            plt.plot(range(n_points), n_points * [0.6])
            plt.savefig(eval_folder + f'/{param_value}/{test}.png')
            plt.clf()

    env.pr.shutdown()


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

    reset_random_seed()

    try:
        eval_folder = eval_folder + '/test_perturbation/'
        os.mkdir(eval_folder)
    except FileExistsError:
        print('Test perturbation exsits, skipping...')
        return

    env, _, agent = build_everything_from_specs(
        specs,
        env_class=CartPolePerturbationEnv,
        exp_path=exp_path
    )

    for _ in tqdm(range(REPEAT_TEST_N_TIMES * 5), desc='Test '
                  'perturbation'):
        BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=STEPS,
            total_steps=STEPS,
            deterministic=True
        )

    env.save_eval(eval_folder=eval_folder)
    env.pr.shutdown()

    df = pd.DataFrame({'recovery_steps': env.mass_state.recover_history})
    print('Recovery steps')
    print(df.describe())
    df.plot.hist()
    plt.savefig(eval_folder + '/recovery_steps')

    df = pd.DataFrame({'unstable_steps': env.mass_state.unstable_history})
    print('Unstables steps')
    print(df.describe())
    df.plot.hist()
    plt.savefig(eval_folder + '/unstable_steps')


def test_noisy_observation(specs, eval_folder: str, exp_path: str):
    """
    Applies random noise to the observations

    Args:
        specs: the specs dict in the experiment folder
        eval_folder: the folder where the evaluation results will be saved
        exp_path: the path to the experiment folder
    """
    STEPS = 750

    reset_random_seed()

    try:
        eval_folder = eval_folder + '/test_noisy_observation/'
        os.mkdir(eval_folder)

    except FileExistsError:
        print('Test noisy_observation exists, skipping...')
        return

    specs['env']['specs']['pomdp_mode'] = 'noise'

    env, _, agent = build_everything_from_specs(
        specs,
        env_class=CartPolePerturbationEnv,
        exp_path=exp_path
    )

    def post_episode_sampling_callback(rewards: List[int]):
        df = pd.DataFrame(env.mass_position_history)
        df.to_csv(eval_folder + 'data.csv', mode='a')

        return rewards

    env.post_episode_sampling_callback = post_episode_sampling_callback

    for _ in tqdm(range(200), desc='Test random noise'):
        BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=STEPS,
            total_steps=STEPS,
            deterministic=True,
            single_episode=True
        )

    env.pr.shutdown()


class NoIncreaseCallback:
    """
    Callback that does not increase env difficulty

    Args:
        env: the CartPoleEnv whose params will be modified
    """

    def __init__(self):
        self.name = self.__class__.__name__

    def __call__(self, env: CartPoleEnv):
        if not hasattr(env, 'repeats'):
            env.repeats = 0
            return 0

        if env.repeats < REPEAT_TEST_N_TIMES:
            env.repeats += 1
            return env.repeats

        return None


class MassIncreaseCallback:
    """
    Callback that increases cartpole mass

    Args:
        env: the CartPoleEnv whose params will be modified
    """

    def __init__(self):
        self.name = self.__class__.__name__
        self.current_mass = None

    def __call__(self, env: CartPoleEnv):
        if self.current_mass is None:
            self.current_mass = env.mass.get_mass()

        if self.current_mass > 1.5:
            return None

        self.current_mass += 0.20

        env.mass.set_mass(self.current_mass)

        return self.current_mass


def read_args() -> Tuple[argparse.Namespace, str, dict]:
    """
    Read the args from the command line and returns it along with the
    eval_folder path and the env specs
    """
    parser = argparse.ArgumentParser(description='Evaluates a trained policy '
                                     'in CartPoleEnv')

    parser.add_argument('--exp_path', type=str, required=True,
                        help='Source path for model binaries')

    args = parser.parse_args()

    experiment_folder = args.exp_path

    with open(experiment_folder + '/specs.json', 'r') as specs_file:
        specs = json.load(specs_file)

    try:
        eval_folder = experiment_folder + '/stats/eval/'
        os.mkdir(eval_folder)

    except FileExistsError:
        pass

    return args, eval_folder, specs


if __name__ == '__main__':
    args, eval_folder, specs = read_args()

    # test_actuation_signal(
    #     eval_folder=eval_folder,
    #     exp_path=args.exp_path,
    #     increase_difficulty_callback=NoIncreaseCallback()
    # )

    # test_actuation_signal(
    #     eval_folder=eval_folder,
    #     exp_path=args.exp_path,
    #     increase_difficulty_callback=MassIncreaseCallback()
    # )

    # test_perturbation(
    #     specs=specs,
    #     eval_folder=eval_folder,
    #     exp_path=args.exp_path
    # )

    # test_noisy_observation(
    #     specs=specs,
    #     eval_folder=eval_folder,
    #     exp_path=args.exp_path
    # )

    test_num_stable_frames(
        specs=specs,
        eval_folder=eval_folder,
        exp_path=args.exp_path
    )
