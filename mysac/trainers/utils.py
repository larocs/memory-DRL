"""
Utils for trainers
"""
import os
import pickle

import torch

from mysac.batch.numpy_batch import NumpySampledBuffer
from mysac.sac.sac import SACAgent


class SaveSACModels:
    """
    Handles SAC checkpointing
    """
    BEST_SCORE = -99999

    @classmethod
    def _save_models(cls, agent: SACAgent, folder: str):
        for model_name in ['policy', 'q1', 'q2', 'q1_target', 'q2_target']:
            model = getattr(agent, model_name)

            torch.save(model, folder + f'/{model_name}.pt')

    @classmethod
    def save_sac_models(cls, agent: SACAgent, experiment_folder: str,
                        score: float):
        """
        Save all models related to the given SACAgent

        Args:
            agent: the SAC Agent
            experiment_folder: the root of the experiment folder
            score: the current model score
        """

        cls._save_models(
            agent=agent,
            folder=experiment_folder + '/models'
        )

        if score > cls.BEST_SCORE:
            print('Saving best: ' + str(score))

            try:
                os.mkdir(path=experiment_folder + '/models/best')
            except FileExistsError:
                pass

            cls.BEST_SCORE = score

            cls._save_models(
                agent=agent,
                folder=experiment_folder + '/models/best'
            )

    @classmethod
    def save_buffer(
        cls,
        buffer: NumpySampledBuffer,
        experiment_folder: str
    ):
        """
        """
        with open(experiment_folder + '/models/buffer', 'wb') as buffer_file:
            pickle.dump(buffer, buffer_file)
