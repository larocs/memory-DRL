"""
Script for plotting SAC Agent statistics as recorded by
`mysac.evaluators.sac_evaluator`
"""
import argparse
import csv

import matplotlib.pyplot as plt

from mysac.evaluators.sac_evaluator import SACEvaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot results for SAC Agent')

    parser.add_argument('--exp_path', type=str,
                        help='Source path for model binaries')
    parser.add_argument('--columns', nargs='*', help='Name of the columns to '
                        'be plot')
    parser.add_argument('--interval', nargs='*', help='Interval to be plotted',
                        type=int)

    args = parser.parse_args()

    columns = args.columns or SACEvaluator.FIELD_NAMES
    values = {name: [] for name in SACEvaluator.FIELD_NAMES}

    with open(args.exp_path + '/stats/sac_stats.csv') as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            for key, value in row.items():
                if '[' in value:
                    value = value[1:-1]

                if key == 'log_prob':
                    value = -float(value)

                values[key].append(float(value))

    start, end = (0, -1)
    if args.interval:
        start, end = args.interval[:2]

    for column_name in columns:
        column_values = values[column_name][start:end]

        plt.plot(
            range(len(column_values)),
            column_values,
            label=column_name
        )

    plt.title(args.exp_path)
    plt.legend()
    plt.show()
