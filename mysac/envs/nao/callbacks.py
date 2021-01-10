from typing import Dict, List

from gym import Env
from mysac.envs.nao import WalkingNao
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler


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
    trajectory = BasicTrajectorySampler.sample_trajectory(
        env=env,
        agent=agent,
        max_steps_per_episode=500,
        total_steps=500,
        deterministic=True,
        single_episode=True
    )

    # We rely on the fact that sample_trajectory will not call reset on the env
    with open(experiment_folder + '/stats/position_history.csv', 'a') as f:
        for point in env.position_history:
            f.write(f'{point[0]};{point[1]}\n')
        f.write(f'-11;-11\n')

    return trajectory
