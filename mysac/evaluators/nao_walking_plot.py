import os
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mysac.envs.nao import RecurrentNAO, WalkingNao
from mysac.evaluators.cartpole_evaluator import read_args, reset_random_seed
from mysac.run_policy import env_from_specs, make_agent, policy_from_specs
from mysac.samplers.sampler import BasicTrajectorySampler
from mysac.envs.nao.constants import STATES_TO_DROP
from tqdm import tqdm

REPEAT_TEST_N_TIMES = 10


def register_forward_walking(
    specs,
    exp_path: str,
    eval_folder: str,
    pomdp_mode: str,
    pomdp_state: str,
    headless: bool = True,
    env_class: callable = WalkingNao,
) -> pd.DataFrame:
    """
    """
    reset_random_seed()

    try:
        eval_folder = eval_folder + '/walking_plot/'
        os.mkdir(eval_folder)

    except FileExistsError:
        print('Test exists, skipping...')
        pass

    specs['env']['specs']['pomdp_mode'] = pomdp_mode
    specs['env']['specs']['pomdp_states'] = [pomdp_state]

    env: WalkingNao = env_from_specs(
        specs=specs,
        headless=headless,
        env_class=env_class,
    )

    policy = policy_from_specs(specs=specs, exp_path=exp_path)

    agent = make_agent(policy=policy, env=env)

    env.action_space.np_random.seed(0)

    position_histories = []

    for _ in tqdm(range(REPEAT_TEST_N_TIMES)):
        BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=250,
            total_steps=250,
            deterministic=True,
            single_episode=True
        )

        position_histories.append(env.position_history[1:])

    env.pr.shutdown()

    history_dict = {
        'episode': [],
        'x': [],
        'y': []
    }
    for episode, episode_history in enumerate(position_histories):
        for x, y in episode_history:
            history_dict['episode'].append(episode)
            history_dict['x'].append(x)
            history_dict['y'].append(y)

    df = pd.DataFrame(history_dict)

    df.to_csv(eval_folder + f'data_{pomdp_mode}_{pomdp_state}.csv')

    return df


def plot(df: pd.DataFrame, title: str, eval_folder: str):
    """
    """
    plt.clf()

    start_x = None
    start_y = None

    for i, episode in enumerate(df.episode.unique()):
        episode_df = df[df.episode == episode]

        if start_x is None:
            start_x = episode_df.x[0]
            start_y = episode_df.y[0]

        plt.plot(episode_df.x, episode_df.y, color=str(0.9 - i * 0.05))

        if episode_df.shape[0] < 250:
            plt.plot(
                episode_df.x.values[-1],
                episode_df.y.values[-1],
                'xr',
                linewidth=2
            )

    plt.plot(
        [start_x, start_x],
        [start_y - 0.15, start_y + 0.15],
        color='orange',
        linewidth=2
    )

    plt.xlabel('x position (m)')
    plt.ylabel('y position (m)')

    plt.title(title)

    plt.savefig(eval_folder + '/walking_plot/' + title)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates a trained policy in CartPoleEnv'
    )

    parser.add_argument('--model_name', type=str, required=True)

    args, eval_folder, specs = read_args(parser=parser)

    if specs['env']['name'] == 'WalkingNao':
        env_class = WalkingNao

    elif specs['env']['name'] == 'RecurrentNAO':
        env_class = RecurrentNAO

    runs = [
        (None, None, 'baseline'),
        ('drop', 'head_z', 'drop head z'),
        ('noise', 'head_z', 'noise head z'),
    ]

    for pomdp_mode, pomdp_state, description in runs:
        df = register_forward_walking(
            specs=specs,
            exp_path=args.exp_path,
            eval_folder=eval_folder,
            env_class=env_class,
            pomdp_state=pomdp_state,
            pomdp_mode=pomdp_mode
        )

        plot(
            df=df,
            title=f'{args.model_name} top-view trajectory ({description})',
            eval_folder=eval_folder
        )
