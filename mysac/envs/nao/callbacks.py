from typing import Dict, List

import numpy as np
from PIL import Image

from mysac.envs.nao import WalkingNao
from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler

NUM_EVAL_STEPS = 10

epochs = 0


def eval_callback(
    agent: SACAgent,
    env: WalkingNao,
    experiment_folder: str,
    save_trajectories_gif: bool = True
) -> Dict[str, List[float]]:
    """
    Evaluation loop for NAO Env

    Args:
        agent: a SAC agent
        env: a Gym Environment
        save_trajectories_gif: record the trajectories and save it as a gif

    Returns:
        A trajectory, as returned by BasicTrajectorySampler.sample_trajectory
    """
    global epochs

    frames: np.array = []

    def step_callback():
        """
        Stores every step as a RGB frame
        """
        frames.append(env.vision_sensor.capture_rgb())

    rewards = []
    for _ in range(NUM_EVAL_STEPS):
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=500,
            total_steps=500,
            deterministic=True,
            single_episode=True,
            step_callback=step_callback if save_trajectories_gif else None
        )

        rewards.append(trajectory['rewards'].sum())

    mean = sum(rewards)/NUM_EVAL_STEPS

    print('Mean eval reward:', mean)
    with open(experiment_folder + '/stats/eval_stats.csv', 'a') as stat_f:
        stat_f.write(f'{mean}\n')

    if save_trajectories_gif:
        images = [
            Image.fromarray((frame * 256).astype(np.uint8))
            for frame in frames
        ]

        images[0].save(
            experiment_folder + f'/stats/eval_{epochs}.gif',
            save_all=True,
            append_images=images[1:],
            duration=100,
            loop=0
        )

    epochs += 1

    return trajectory
