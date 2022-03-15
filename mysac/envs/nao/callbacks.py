from typing import Dict, List

import numpy as np
from mysac.envs.nao import WalkingNao
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler

NUM_EVAL_STEPS = 10


def eval_callback(
    agent: SACAgent, env: WalkingNao, experiment_folder: str) \
        -> Dict[str, List[float]]:
    """
    Evaluation loop for NAO Env

    Args:
        agent: a SAC agent
        env: a Gym Environment

    Returns:
        A trajectory, as returned by BasicTrajectorySampler.sample_trajectory
    """
    rewards = []
    for _ in range(NUM_EVAL_STEPS):
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=500,
            total_steps=500,
            deterministic=True,
            single_episode=True
        )

        rewards.append(trajectory['rewards'].sum())

    mean = sum(rewards)/NUM_EVAL_STEPS

    print('Mean eval reward:', mean)
    with open(experiment_folder + '/stats/eval_stats.csv', 'a') as stat_f:
        stat_f.write(f'{mean}\n')

    return trajectory
