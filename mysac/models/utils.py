import numpy as np
import torch


def fanin_init(tensor: torch.tensor):
    """ Initializes a MLP weigths in-place """
    size = tensor.size()

    if len(size) == 2:
        fan_in = size[0]

    elif len(size) > 2:
        fan_in = np.prod(size[1:])

    else:
        raise Exception("Shape must be have dimension at least 2.")

    bound = 1. / np.sqrt(fan_in)

    return tensor.data.uniform_(-bound, bound)
