from email.charset import BASE64
from os import path

import pandas as pd

TEST_TYPES = [
    'random',
    'zero'
]

STATE_SUFIX = [
    'angular',
    'linear',
    'head_z',
    'joint_positions',
    'orientation_x',
    'orientation_y',
    'orientation_z'
]

BASE_PATH = 'experiments/walking_nao_512_delayed_higher_energy_cost_lstm_gpu_again/stats/eval/'

# if __name__ == '__main__':
for test_type in TEST_TYPES:
    print(test_type.capitalize() + ':')

    for state_sufix in STATE_SUFIX:
        csv = pd.read_csv(
            path.join(
                BASE_PATH,
                test_type,
                f'test_pomdp_{state_sufix}',
                'data.csv'
            )
        )

        print(
            '-',
            state_sufix.capitalize() + ':',
            f'{csv.steps.mean():.2f} ±{csv.steps.std():.2f}'
        )
