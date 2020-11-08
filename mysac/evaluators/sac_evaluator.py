"""
This module contains code for collecting SAC metrics during training
"""
import csv
from typing import List

import torch


class SACEvaluator:
    """
    Aggregates and saves metrics for the SAC Agent.
    The csv file will reside in `experiment_path` + '/stats/sac_stats.csv'.

    Args:
        experiment_path: the path to the experiment root folder
    """

    def __init__(self, experiment_path: str):
        self.log_probs: List[float] = []
        self.q1_losses: List[float] = []
        self.q2_losses: List[float] = []
        self.policy_losses: List[float] = []
        self.alphas: List[float] = []

        self._calls = 0
        self.experiment_path = experiment_path
        self.stats_file_path = experiment_path + '/stats/sac_stats.csv'

        self.field_names = ['log_prob', 'q1_loss', 'q2_loss', 'policy_loss',
                            'alpha']

        with open(self.stats_file_path, 'a+', newline='') as stat_fp:
            dict_writer = csv.DictWriter(stat_fp, fieldnames=self.field_names)
            dict_writer.writeheader()

    def aggregate_values(self, log_prob: torch.tensor, q1_loss: torch.tensor,
                         q2_loss: torch.tensor, policy_loss: torch.tensor,
                         alpha: torch.tensor):
        """
        Receives lists of tensors containing information about the SAC backward
        steps and keeps it buffered until `save_metrics` is called.
        """
        self.log_probs.append(log_prob.mean().numpy())
        self.q1_losses.append(q1_loss.numpy())
        self.q2_losses.append(q2_loss.numpy())
        self.policy_losses.append(policy_loss.numpy())
        self.alphas.append(alpha.numpy())

        self._calls += 1

    def save_metrics(self):
        """
        Persists the metrics aggregated by `aggregate_values` into a csv file
        """
        with open(self.stats_file_path, 'a+', newline='') as stat_fp:
            dict_writer = csv.DictWriter(stat_fp, fieldnames=self.field_names)

            rows = [{
                'log_prob': self.log_probs[i],
                'q1_loss': self.q1_losses[i],
                'q2_loss': self.q2_losses[i],
                'policy_loss': self.policy_losses[i],
                'alpha': self.alphas[i]
            } for i in range(len(self.log_probs))]

            dict_writer.writerows(rows)

        self.log_probs = []
        self.q1_losses = []
        self.q2_losses = []
        self.policy_losses = []
        self.alphas = []
