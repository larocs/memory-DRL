import torch


def rescale_action(self, action):
    pass


def update_target_network(q_model: torch.nn.Module, q_target: torch.nn.Module,
                          tau: float):
    """ Updates q_model weights from q_target given a tau in-place. """
    ziped_params = zip(q_model.parameters(), q_target.parameters())

    for param, target_param in ziped_params:
        target_param.data.copy_(
            param.data * tau + (1 - tau) * target_param.data
        )
