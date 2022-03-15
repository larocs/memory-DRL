from copy import deepcopy

from mysac.envs.nao import RecurrentNAO, WalkingNao
from mysac.envs.nao.constants import STATES_TO_DROP
from mysac.evaluators.cartpole_evaluator import read_args
from mysac.evaluators.nao_walking_evaluator import evaluate_forward_walking

if __name__ == '__main__':
    args, eval_folder, specs = read_args()

    if specs['env']['name'] == 'WalkingNao':
        env_class = WalkingNao

    elif specs['env']['name'] == 'RecurrentNAO':
        env_class = RecurrentNAO

    for state_to_drop in STATES_TO_DROP.keys():
        pomdp_specs = deepcopy(specs)
        pomdp_specs['env']['specs']['pomdp_states'] = [state_to_drop]
        pomdp_specs['env']['specs']['pomdp_mode'] = 'noise'

        evaluate_forward_walking(
            specs=pomdp_specs,
            exp_path=args.exp_path,
            eval_folder=eval_folder,
            env_class=env_class,
            test_name=f'/random/test_pomdp_{state_to_drop}/'
        )
