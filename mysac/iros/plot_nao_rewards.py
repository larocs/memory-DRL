from os import path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from mysac.iros.plot_cartpole_rewards import moving_average

EXP_PATHS = {
    'walking_nao_512_delayed_higher_energy_cost_lstm_gpu_again': 'LSTM',
    'walking_nao_1024_delayed_energy_cost': 'FCN'
}

COLORS = [
    ('#80b7d1', '#003f5c'),
    ('#b4b0d4', '#58508d')
]

if __name__ == '__main__':
    iterator = zip(EXP_PATHS.items(), COLORS)

    for (exp_path, exp_name), (bg_color, fg_color) in iterator:
        df = pd.read_csv(
            path.join(
                'experiments/',
                exp_path,
                'stats/eval_stats.csv',
            ),
            names=['values']
        )

        sns.set_style("whitegrid")

        # plt.plot(df.values[:2000], color=bg_color)

        plt.plot(
            moving_average(df.values[:2000]),
            label=exp_name,
            color=fg_color
        )

    plt.ylim([0, 11000])

    plt.title(f'Training evaluation for NAO')
    plt.ylabel('Average reward')
    plt.xlabel('Evaluation steps')
    plt.legend()
    plt.tight_layout()
    plt.show()
