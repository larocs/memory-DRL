import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mysac.envs.nao import RecurrentNAO, WalkingNao
from mysac.evaluators.cartpole_evaluator import read_args, reset_random_seed
from mysac.run_policy import env_from_specs, make_agent, policy_from_specs
from mysac.samplers.sampler import BasicTrajectorySampler
from tqdm import tqdm
from typing_extensions import final

REPEAT_TEST_N_TIMES = 10


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

    df = pd.DataFrame(
        {
            'deslocation': deslocations,
            'steps': total_steps
        }
    )

    df['velocity'] = df.deslocation / df.steps
    print(df.describe)
    df.to_csv(eval_folder + 'data.csv')

    plt.title('Delayed Energy Cost for the last test')
    plt.plot(env.delayed_energy_cost)
    plt.savefig(eval_folder + 'delayed_energy_cost.png')

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
