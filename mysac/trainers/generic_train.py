from typing import Callable, Dict, List

import numpy as np
import torch
from gym import Env
from tqdm import tqdm

from mysac.sac.sac import SACAgent
from mysac.samplers.sampler import BasicTrajectorySampler

from .utils import SaveSACModels

EvalCallback = Callable[[SACAgent, Env], Dict[str, List[float]]]


def default_eval_callback(
    agent: SACAgent, env: Env, experiment_folder: str) \
        -> Dict[str, List[float]]:
    """
    Default evaluation loop, implemented for backward compability

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
        deterministic=True
    )

    print('Eval reward:', np.sum(trajectory['rewards']))
    with open(experiment_folder + '/stats/eval_stats.csv', 'a') as stat_f:
        stat_f.write(f'{np.sum(trajectory["rewards"])}\n')

    return trajectory


def generic_train(
        env: Env,
        agent: Callable,
        buffer: Callable,
        experiment_folder: str,
        batch_size: int = 256,
        max_steps_per_episode: int = 250,
        sampled_steps_per_epoch: int = 1000,
        train_steps_per_epoch: int = 1000,
        evaluator: Callable[[Dict[str, torch.tensor]], None] = None,
        num_epochs: int = int(1e6),
        eval_callback: EvalCallback = default_eval_callback):
    """ Generic, infinite train loop with deterministic evaluation.

    Args:
        agent: the agent interface (should expose a get_action interface)
        buffer: the data structure to keep the replay buffer
        experiment_folder: path to the experiment folder where the models and
            evaluation data will be saved
        batch_size: the size of batch that will be sampled from the replay
            buffer at each train step
        max_steps_per_episode: the max steps per episode during sample
        sampled_steps_per_epoch: how many steps should be sampled at each
            epoch
        train_steps_per_epoch: how many backward passes per epoch
        num_epochs: the number of epochs for training
    """
    for current_epoch in range(num_epochs):
        # Eval
        eval_trajectory = eval_callback(
            agent=agent,
            env=env,
            experiment_folder=experiment_folder
        )

        # Sample
        trajectory = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=max_steps_per_episode,
            total_steps=sampled_steps_per_epoch
        )

        buffer.add_trajectory(**trajectory)

        for _ in tqdm(range(train_steps_per_epoch)):
            batch = buffer.sample(batch_size)

            agent_info: Dict[str, torch.tensor] = agent \
                .train_from_samples(batch=batch)

            if evaluator:
                evaluator.aggregate_values(**agent_info)

        score = np.sum(eval_trajectory['rewards'])

        SaveSACModels.save_sac_models(
            agent=agent,
            experiment_folder=experiment_folder,
            score=score,
            epoch=current_epoch if score > 8_000 else None
        )

        SaveSACModels.save_buffer(
            buffer=buffer,
            experiment_folder=experiment_folder
        )

        if evaluator:
            evaluator.save_metrics()

        if score > 8_750 and score <= 8_900:
            print('Ending on score', score)
            break

    print('CABO O LOOP')
