# pylint: disable=no-member
# pylint: disable=not-callable

import argparse
import json

import torch
from gym import Env
from mysac.envs.cartpole_ignore_inputs import CartPoleIgnoreStatesEnv
from mysac.envs.nao import WalkingNao
from mysac.envs.pyrep_env import CartPoleEnv
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler


def env_from_specs(specs, env_class: Env = CartPoleEnv,
                   headless: bool = True) -> Env:
    """
    Returns a CartPoleEnv from a specs json

    Args:
        specs: the specs in a json-like structure
        headless: overrides the headless option in the specs dict with this
            param value
    """
    env_specs = specs['env']['specs']

    buffer_for_rnn = env_specs.get('buffer_for_rnn', None)
    if buffer_for_rnn is None:
        env_specs['buffer_for_rnn'] = False

    env_specs['headless'] = headless

    return env_class(**env_specs)


def get_env_class(env_name: str) -> Env:
    """
    Returns the correct Env class given a name
    """
    if env_name == 'CartPole':
        return CartPoleEnv

    if env_name == 'CartPoleIgnoreStatesEnv':
        return CartPoleIgnoreStatesEnv

    if env_name == 'WalkingNao':
        return WalkingNao

    raise ValueError('The env required in specs is not recognized')


def policy_from_specs(specs, exp_path: str) -> torch.nn.Module:
    """
    Loads a Policy Network from a specs json-like

    Args:
        specs: the specs in a json-like structure
        exp_path: the path to the experiment where the policy is saved
    """
    # Select the model
    if specs['models']['mode'] == 'rnn':
        from mysac.models.rnn_models import PolicyModel, QModel

    else:
        from mysac.models.mlp import PolicyModel, QModel

    policy = torch.load(exp_path + '/models/policy.pt')
    policy.eval()

    return policy


def make_agent(policy: torch.nn.Module, env: Env) -> SACAgent:
    """
    Returns a SACAgent instance to the given policy

    Args:
        policy: the policy network
        env: an Gym-like env
    """
    return SACAgent(
        env=env,
        policy_model=policy,
        # We don't care about the other networks
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run a trained policy from '
                                     'specs')

    parser.add_argument('--exp_path', type=str,
                        help='Source path for model binaries')

    args = parser.parse_args()

    experiment_folder = args.exp_path

    with open(experiment_folder + '/specs.json', 'r') as specs_file:
        specs = json.load(specs_file)

    print('Specs:', specs)
    env_class = get_env_class(env_name=specs['env']['name'])
    policy = policy_from_specs(specs=specs, exp_path=args.exp_path)
    env = env_from_specs(specs=specs, headless=False, env_class=env_class)
    agent = make_agent(policy=policy, env=env)

    BasicTrajectorySampler.sample_trajectory(
        env=env,
        agent=agent,
        max_steps_per_episode=250,
        total_steps=1e6,
        deterministic=True
    )
