import argparse
import importlib
import json
import shutil
import subprocess
from datetime import datetime
from os import mkdir, path
from typing import Dict

import numpy
import torch

from mysac.batch.numpy_batch import (NumpySampledBuffer,
                                     NumpySampledBufferForRNN)
from mysac.envs.cartpole_ignore_inputs import CartPoleIgnoreStatesEnv
from mysac.envs.marta.marta import MartaWalkEnv
from mysac.envs.nao import RecurrentNAO, WalkingNao
from mysac.envs.pyrep_env import CartPoleEnv
from mysac.evaluators.sac_evaluator import SACEvaluator
from mysac.sac.sac import SACAgent
from mysac.trainers.generic_train import generic_train
from mysac.utils import get_device


def create_folders(experiment_folder: str):
    """
    Create the folder structure for the experiment

    Args:
        experiment_folder: the path where the folder structure will be created

    Raises:
        FileExistsError: if the structure already exists
    """
    if path.isdir(experiment_folder + "/models"):
        confirmation = input("Experiment already exists! Override? [y/N]: ")

        if confirmation == "n":
            raise ValueError(
                "Pick a different folder for the experiment " "artifacts")

        else:
            shutil.rmtree(experiment_folder + "/models/", ignore_errors=True)
            shutil.rmtree(experiment_folder + "/stats/", ignore_errors=True)

    mkdir(experiment_folder + "/models/")
    mkdir(experiment_folder + "/stats/")


def run_experiment_from_specs(experiment_folder: str):
    """ Run an experiment from a dictionaty of specifications

    Args:
        experiment_folder: the path to the experiment folder. If does not
            exists, it is created. If it already exists, we try to load a file
            with specs within it. """
    create_folders(experiment_folder)

    meta = {
        "branch": subprocess.check_output(["git", "branch"]).decode(),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]
        ).decode(),
        "date": datetime.now().strftime("%m/%d/%Y, %H:%M:%S"),
    }

    with open(experiment_folder + "/specs.json", "r") as specs_file:
        specs = json.load(specs_file)

    # Fixes the random seeds
    torch.manual_seed(specs["seed"])
    numpy.random.seed(specs["seed"])

    # Select the model
    if specs["models"]["mode"] == "rnn":
        from mysac.models.rnn_models import PolicyModel, QModel

        buffer = NumpySampledBufferForRNN(**specs["buffer"])

    elif specs["models"]["mode"] == "mlp":
        from mysac.models.mlp import PolicyModel, QModel

        buffer = NumpySampledBuffer(**specs["buffer"])

    elif specs["models"]["mode"] == "attention":
        from mysac.models.attention_model import PolicyModel, QModel

        buffer = NumpySampledBufferForRNN(**specs["buffer"])

    elif specs["models"]["mode"] == "crazy":
        from mysac.models.crazy_model import PolicyModel, QModel

        buffer = NumpySampledBufferForRNN(**specs["buffer"])

    device = get_device()
    policy = PolicyModel(**specs["models"]["policy"]).to(device)
    q1_model = QModel(**specs["models"]["q_model"]).to(device)
    q1_target = QModel(**specs["models"]["q_model"]).to(device)
    q2_model = QModel(**specs["models"]["q_model"]).to(device)
    q2_target = QModel(**specs["models"]["q_model"]).to(device)

    print("Policy:", policy)

    env_name = specs["env"]["name"]

    if env_name == "CartPole":
        env = CartPoleEnv(**specs["env"]["specs"])

    elif env_name == "CartPoleIgnoreStatesEnv":
        env = CartPoleIgnoreStatesEnv(**specs["env"]["specs"])

    elif env_name == "WalkingNao":
        env = WalkingNao(**specs["env"]["specs"])

    elif env_name == "RecurrentNAO":
        env = RecurrentNAO(**specs["env"]["specs"])

    elif env_name == 'MartaWalkEnv':
        env = MartaWalkEnv(**specs["env"]["specs"])

    agent = SACAgent(
        # Env
        env=env,
        # Models
        policy_model=policy,
        q1_model=q1_model,
        q2_model=q2_model,
        q1_target=q1_target,
        q2_target=q2_target,
        # Hyperparams
        **specs["hyperparams"]
    )

    with open(experiment_folder + "/meta.json", "w") as meta_file:
        json.dump(meta, meta_file)

    try:
        generic_train(
            env=env,
            agent=agent,
            buffer=buffer,
            experiment_folder=experiment_folder,
            evaluator=SACEvaluator(experiment_folder),
            **callback_loader(specs=specs),
            **specs["trainer"]
        )

    except KeyboardInterrupt:
        meta["end_date"] = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

        with open(experiment_folder + "/meta.json", "w") as meta_file:
            json.dump(meta, meta_file)


def callback_loader(specs: Dict) -> Dict[str, callable]:
    """
    Load callbacks from the specs file

    Args:
        specs: json-like, representing a train spec file

    Returns:
        A dictionary mapping from callback names to callback functions
    """
    if "callbacks" not in specs:
        return {}

    return {
        name: import_function(function_path=path)
        for name, path in specs["callbacks"].items()
    }


def import_function(function_path: str) -> callable:
    """
    Returns a function given its path

    Args:
        function_path: the path to the function
    """
    *module_path, function_name = function_path.split(".")
    module_path = ".".join(module_path)

    module_path = importlib.import_module(name="".join(module_path))

    return getattr(module_path, function_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SAC from specifications")

    parser.add_argument(
        "--exp_path", type=str, help="Output path for model binaries and stats"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Run the deterministic algorithm"
    )
    parser.add_argument("--viz", action="store_true")

    args = parser.parse_args()

    if not args.deterministic:
        run_experiment_from_specs(args.exp_path)
