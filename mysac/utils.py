import torch


def get_device():
    """
    Returns the device to be used by torch
    """
    if torch.cuda.is_available():
        device_name = "cuda:0"
    else:
        device_name = "cpu"

    return torch.device(device_name)
