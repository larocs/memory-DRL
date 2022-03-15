import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mysac.envs.nao import RecurrentNAO, WalkingNao
from mysac.evaluators.cartpole_evaluator import read_args, reset_random_seed
from mysac.run_policy import env_from_specs, make_agent, policy_from_specs
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm

REPEAT_TEST_N_TIMES = 200


def evaluate_forward_walking(
        specs,
        exp_path: str,
        eval_folder: str,
        headless: bool = True,
        env_class: callable = WalkingNao):
    """
    """
    reset_random_seed()

    try:
        eval_folder = eval_folder + '/test_forward_walking/'
        os.mkdir(eval_folder)

    except FileExistsError:
        print('Test exsits, skipping...')
        return

    env: WalkingNao = env_from_specs(
        specs=specs,
        headless=headless,
        env_class=env_class,
    )

    policy = policy_from_specs(specs=specs, exp_path=exp_path)

    agent = make_agent(policy=policy, env=env)

    env.action_space.np_random.seed(0)

    deslocations = []
    total_steps = []
    start_x, start_y, _ = env.chest.get_position()

    position_histories = []

    for _ in tqdm(range(REPEAT_TEST_N_TIMES)):
        results = BasicTrajectorySampler.sample_trajectory(
            env=env,
            agent=agent,
            max_steps_per_episode=250,
            total_steps=250,
            deterministic=True,
            single_episode=True
        )

        final_x, final_y, _ = env.chest.get_position()

        total_steps.append(len(results['rewards']))
        deslocations.append(
            np.linalg.norm(
                np.array((start_x, start_y)) - np.array((final_x, final_y))
            )
        )

        position_histories.append(env.position_history[1:])

    x_desloc = []
    y_residual = []
    for position_history in position_histories:
        positions = np.array(position_history)

        y_residual.append(np.abs(positions[:, 1] - -1.1950e+01).sum())

        x_desloc.append(np.abs(positions[0, 0] - positions[-1, 0]))

    df = pd.DataFrame(
        {
            'deslocation': deslocations,
            'steps': total_steps,
            'y_residual': y_residual,
            'x_desloc': x_desloc
        }
    )

    df['velocity'] = df.deslocation / df.steps
    print(df.describe)
    df.to_csv(eval_folder + 'data.csv')

    plt.title('Delayed Energy Cost for the last test')
    plt.plot(env.delayed_energy_cost)
    plt.savefig(eval_folder + 'delayed_energy_cost.png')

    plt.clf()
    plt.title('Estatisticas de deslocamento')
    plt.plot(
        df['y_residual'],
        label='Soma dos residuos de Y por episódio',
        marker='o'
    )
    plt.plot(
        df['x_desloc'],
        label='Deslocamento em x',
        marker='o'
    )
    plt.xlabel('Episódio')
    plt.savefig(eval_folder + 'desloc_stats.png')

    for i, position_history in enumerate(position_histories):
        position_history = np.array(position_history)
        x, y = position_history[:, 0], position_history[:, 1]

        plt.clf()
        plt.title(f'Run {i}')
        plt.quiver(
            x[:-1], y[:-1],
            x[1:]-x[:-1], y[1:]-y[:-1],
            scale_units='xy',
            angles='xy',
            scale=1
        )

        plt.xlabel('Posição em X')
        plt.ylabel('Posição em Y')

        plt.savefig(eval_folder + f'run_{i}.png')

    env.pr.shutdown()


if __name__ == '__main__':
    args, eval_folder, specs = read_args()

    if specs['env']['name'] == 'WalkingNao':
        env_class = WalkingNao

    elif specs['env']['name'] == 'RecurrentNAO':
        env_class = RecurrentNAO

    evaluate_forward_walking(
        specs=specs,
        exp_path=args.exp_path,
        eval_folder=eval_folder,
        env_class=env_class
    )
