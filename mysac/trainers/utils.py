"""
Utils for trainers
"""
import torch
from mysac.sac.sac import SACAgent


def save_sac_models(agent: SACAgent, experiment_folder: str):
    """
    Save all models related to the given SACAgent

    Args:
        agent: the SAC Agent
        experiment_folder: the root of the experiment folder
    """

    for model_name in ['policy', 'q1', 'q2', 'q1_target', 'q2_target']:
        model = getattr(agent, model_name)

        torch.save(model, experiment_folder + f'/models/{model_name}.pt')
