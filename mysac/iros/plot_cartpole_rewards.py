import re
from collections import OrderedDict
from os import path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

CARTPOLE_EXPS_NONOISE = OrderedDict({
    'cartpole_selfatt_fork': 'Self-attention',
    'cartpole_2linear_2': 'LSTM',
    'fc_test_remake': 'Fully Connected Network'
})

CARTPOLE_EXPS_WNOISE = OrderedDict({
    'cartpole_selfatt_fork_noise': 'Self-attention',
    'cartpole_2linear_noise': 'LSTM',
    'fc_test_noise': 'Fully Connected Network',
})

COLORS = [
    ('#edb7d7', '#bc5090'),
    ('#80b7d1', '#003f5c'),
    ('#b4b0d4', '#58508d')
]


def moving_average(a: np.array, n: int = 10):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def cut_on_max(array: np.array) -> np.array:
    return array[:np.argmax(array) + 1]


def plot_model_rewards(
        exp_paths: Dict[str, str], title: str, filename: str,
        ignore: Optional[List[str]] = None
):
    plt.clf()

    iterator = zip(exp_paths.items(), COLORS)

    for (experiment_path, experiment_name), (bg_color, fg_color) in iterator:
        if ignore and experiment_name in ignore:
            continue

        stats_path = path.join(
            'experiments',
            experiment_path,
            'stats',
            'eval_stats.csv'
        )

        values = pd.read_csv(stats_path, header=None, names=['values']).values

        sns.set_style("whitegrid")

        # plt.plot(
        #     values[:5800],
        #     color=bg_color
        # )

        plt.plot(
            moving_average(values[:5800], n=50),
            label=experiment_name,
            color=fg_color
        )

    plt.title(title)
    plt.ylabel('Average reward')
    plt.xlabel('Evaluation steps')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename)


if __name__ == '__main__':
    plot_model_rewards(
        exp_paths=CARTPOLE_EXPS_NONOISE,
        title='Training evaluation for Cartpole (trained without noise)',
        filename='training_cartpole_withoutnoise'
    )

    plot_model_rewards(
        exp_paths=CARTPOLE_EXPS_WNOISE,
        title='Training evaluation for Cartpole (trained with noise)',
        filename='training_cartpole_withnoise',
        ignore=['Self-attention']
    )
