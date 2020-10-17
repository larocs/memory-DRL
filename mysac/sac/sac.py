import torch
from gym import Env


class SACAgent:
    def __init__(self,
                 env: Env,
                 agent_model_class: torch.nn.Module,
                 q_model_class: torch.nn.Module,
                 gamma: float,
                 agent_lr: float,
                 q_lr: float):

        self.agent = agent_model_class()
        self.q1 = q_model_class()

    def get_action(self, deterministic: bool = False):
        return
