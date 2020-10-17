from gym import Env
from typing import Tuple, List


class BasicTrajectorySampler:
    @staticmethod
    def sample_trajectory(env: Env, agent, max_steps_per_episode: int,
                          total_steps: int,
                          deterministic: bool = False) -> Tuple[List[float], List[float],
                                                                List[float], List[float],
                                                                List[float]]:
        """ Sample a trajectory

        Args:
            env: the environment where the steps will be sampled from
            agent: the agent for sampling the actions given the observations
            max_steps_per_episode: the max number of steps per episode
            total_steps: the number of steps to be collected from the env
            deterministic: if True, use the best actions only

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
        done = False

        while steps < total_steps:
            observation = env.reset()

            episode_steps = 0

            while not done and episode_steps < max_steps_per_episode \
                    and steps < total_steps:
                action = agent.get_action(
                    observations=observation,
                    reparametrize=False
                )

                next_observation, reward, done, _ = env.step()

                observations.append(observation)
                next_observations.append(next_observation)
                actions.append(action)
                rewards.append(reward)
                terminals.append(done)

                observation = next_observation

                steps += 1
                episode_steps += 1

            episodes += 1

        return (
            observations,
            next_observations,
            rewards,
            actions,
            terminals
        )
