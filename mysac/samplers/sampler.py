from typing import Dict, List

import numpy as np
import torch
from gym import Env
from mysac.utils import get_device


class BasicTrajectorySampler:
    @staticmethod
    def sample_trajectory(
            env: Env, agent, max_steps_per_episode: int,
            total_steps: int,
            deterministic: bool = False,
            single_episode: bool = False,
            reset_kwargs: dict = None) -> Dict[str, List[float]]:
        """ Sample a trajectory

        Args:
            env: the environment where the steps will be sampled from
            agent: the agent for sampling the actions given the observations
            max_steps_per_episode: the max number of steps per episode
            total_steps: the number of steps to be collected from the env
            deterministic: if True, use the best actions only
            single_episode: if True, will sample untill max_steps or the first
                env reset, whichever comes first
            reset_kwargs: a dict of kwargs to be passed to the env.reset()
                method

        Returns:
            5 lists of float values: observations, next observations, rewards,
                actions and terminals
        """
        # Keept the trajectory
        observations = []
        next_observations = []
        rewards = []
        actions = []
        terminals = []

        # Count the total steps and episodes
        steps = 0
        episodes = 0

        if not reset_kwargs:
            reset_kwargs = {}

        while steps < total_steps:
            episode_rewards = []
            observation = env.reset(**reset_kwargs)
            episode_steps = 0
            done = False

            while not done and episode_steps < max_steps_per_episode \
                    and steps < total_steps:
                action, *_ = agent.get_action(
                    observations=torch.tensor(
                        observation, device=get_device()),
                    reparametrize=False,
                    deterministic=deterministic
                )

                action = action.detach().cpu().numpy()

                next_observation, reward, done, _ = env.step(action)

                observations.append(observation)
                next_observations.append(next_observation)
                actions.append(action)
                episode_rewards.append(np.array([reward]))
                terminals.append(np.array([int(done)]))

                observation = next_observation

                steps += 1
                episode_steps += 1

            if hasattr(env, 'post_episode_sampling_callback'):
                episode_rewards = env.post_episode_sampling_callback(
                    rewards=episode_rewards
                )

            rewards.extend(episode_rewards)

            episodes += 1

            if single_episode:
                break

        return {
            'observations': np.array(observations),
            'next_observations': np.array(next_observations),
            'rewards': np.array(rewards),
            'actions': np.array(actions),
            'terminals': np.array(terminals)
        }
