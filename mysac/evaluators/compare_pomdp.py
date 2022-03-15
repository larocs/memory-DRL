import argparse
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PATH_TO_CSV = 'stats/eval/{noise_type}/test_pomdp_{state}/track.csv'
Y_CENTER = -1.1950e+01


def open_csv(
        exp_path: str, droped_state: str, noise_type: str) -> pd.DataFrame:
    """
    Opens path tracking data and returns it as a CSV

    Args:
        exp_path: path to the root directory of the experiment
        droped_state: name of the droped state
        noise_type: one of zero/random
    """
    sufix = PATH_TO_CSV.format(state=droped_state, noise_type=noise_type)

    with open(path.join(exp_path, sufix)) as data_file:
        return pd.read_csv(data_file)


def compute_y_error_per_x_desloc(track_df: pd.DataFrame) -> float:
    x = track_df.x.to_numpy()
    y = track_df.y.to_numpy()

    y = np.abs(y - Y_CENTER).sum()

    return y / np.abs(x[-1] - x[0])


def plot_hist(args: argparse.ArgumentParser):
    results = []
    for exp_path in [args.exp_a_path, args.exp_b_path]:
        track_df = open_csv(
            exp_path=exp_path,
            droped_state=args.droped_state
        )

        exp_results = [
            compute_y_error_per_x_desloc(
                track_df=track_df[track_df.run == run_ix]
            )

            for run_ix in track_df.run.unique()
        ]

        results.append(exp_results)

    plt.hist(
        results,
        label=[
            args.exp_a_path,
            args.exp_b_path
        ]
    )

    plt.title('Observalidade parcial em ' + args.droped_state)
    plt.xlabel('Erro acumulado em Y')
    plt.ylabel('Número de ocorrências')

    plt.legend()
    plt.show()


def plot_scatter(args: argparse.ArgumentParser):
    for exp_path in [args.exp_a_path, args.exp_b_path]:
        track_df = open_csv(
            exp_path=exp_path,
            droped_state=args.droped_state,
            noise_type=args.noise_type
        )

        print(exp_path)

        print(
            'runs com > 240 steps:',
            (track_df.run.value_counts() > 240).sum()
        )

        print('média de steps:', track_df.run.value_counts().mean())
        print()

        x_desloc = []
        y_error = []
        for run_ix in track_df.run.unique():
            run_df = track_df[track_df.run == run_ix]

            x = run_df.x.to_numpy()[1:]
            y = run_df.y.to_numpy()[1:]

            # print('[' + ','.join(str(y_) for y_ in y) + ']')

            x_desloc.append(x[-1] - x[0])
            y_error.append(np.abs(y - Y_CENTER).sum())

        plt.scatter(x_desloc, y_error, label=exp_path)

    plt.title('Observalidade parcial em ' + args.droped_state)
    plt.ylabel('Erro acumulado em Y')
    plt.xlabel('Deslocamento em X')

    plt.legend()
    plt.show()


def plot_correlation(args: argparse.ArgumentParser):
    import seaborn as sb

    for exp_path in [args.exp_a_path]:
        track_df = open_csv(
            exp_path=exp_path,
            droped_state=args.droped_state
        )

        track_df.drop(columns=['Unnamed: 0'])

        x_desloc = []
        y_error = []
        n_steps = []
        for run_ix in track_df.run.unique():
            run_df = track_df[track_df.run == run_ix]

            x = run_df.x.to_numpy()[1:]
            y = run_df.y.to_numpy()[1:]

            # print('[' + ','.join(str(y_) for y_ in y) + ']')

            x_desloc.append(x[-1] - x[0])
            y_error.append(np.abs(y - Y_CENTER).sum())
            n_steps.append(len(x_desloc))

        corr_df = pd.DataFrame(
            {
                'x_desloc': x_desloc,
                'y_error': y_error,
                'n_steps': n_steps
            }
        )

        sb.heatmap(corr_df.corr(), cmap="Blues", annot=True)
        plt.show()


def plot_corelation_2():
    from os import path

    tracks = [
        'test_pomdp_angular',
        'test_pomdp_linear',
        'test_pomdp_orientation_z',
        'test_pomdp_head_z',
        'test_pomdp_orientation_x',
        'test_pomdp_joint_positions',
        'test_pomdp_orientation_y'
    ]

    base_path = '~/Develop/IC/sac_experiments/experiments/'

    exp_a = 'walking_nao_512_delayed_higher_energy_cost_lstm_gpu_again'
    exp_b = 'walking_nao_1024_delayed_energy_cost'

    exp_success = {
        exp: {}
        for exp in [exp_a, exp_b]
    }

    for exp in [exp_a, exp_b]:
        for track in tracks:
            track_df = pd.read_csv(
                path.join(base_path, exp, 'stats/eval', track, 'track.csv'))

            n_success = (track_df.run.value_counts() > 240).sum()

            exp_success[exp][' '.join(track.split('_')[2:])] = n_success

    print(pd.DataFrame(exp_success))


def plot_cartpole_scatter(args: argparse.ArgumentParser):
    """
    """
    for exp_path in [args.exp_a_path, args.exp_b_path]:
        track_df = pd.read_csv(
            exp_path,
            names=['ep', 'x', 'y', 'z']
        )

        track_df.columns = ['ep', 'x', 'y', 'z']
        track_df.z = (track_df.z > 0.55).astype(int)

        print(
            exp_path,
            '- Média de Steps com Z > 0.5 para 200 runs (max steps: 750):',
            track_df.groupby('ep').sum().z.mean()
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compares two POMDP runs')

    parser.add_argument('exp_a_path', type=str, help='First exp to compare')

    parser.add_argument('exp_b_path', type=str, help='Second exp to compare')

    parser.add_argument('droped_state', type=str,
                        help='Droped state to compare')

    parser.add_argument('noise_type', type=str, help='The type of noise',
                        choices=['zero', 'random', 'baseline'])

    args = parser.parse_args()

    # plot_hist()
    # plot_scatter(args)
    # plot_correlation(args)
    # plot_corelation_2()
    plot_cartpole_scatter(args=args)
