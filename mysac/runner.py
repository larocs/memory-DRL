import json
from os import mkdir, path
import argparse


def run_experiment_from_specs(experiment_folder: str):
    """ Run an experiment from a dictionaty of specifications

    Args:
        experiment_folder: the path to the experiment folder. If does not
            exists, it is created. If it already exists, we try to load a file
            with specs within it. """
    try:
        mkdir(experiment_folder)

    except FileExistsError:
        with open(path.join(experiment_folder, 'specs.json')) as specs_file:
            train_specs = json.load(specs_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run sac for CartPole2d')

    parser.add_argument('--exp_path', type=str,
                        help='Output path for model binaries and stats')
    parser.add_argument('--deterministic', action='store_true',
                        help='Run the deterministic algorithm')
    parser.add_argument('--viz', action='store_true')

    args = parser.parse_args()

    if not args.deterministic:
        run_experiment_from_specs(args.exp_path)
