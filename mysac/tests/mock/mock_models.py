""" Module for creating and loading mocked models whose weights"""
import torch

# TODO: makes it usable across test modules
DEFAULT_OBS_SIZE = 2
DEFAULT_ACTION_SIZE = 3
DEFAULT_HIDDEN_SIZE = 5

PATH = './mocked_models/'


class MockedQmodel(torch.nn.Module):
    """ Mocked QModel """

    def __init__(self):
        super().__init__()

        self.layer = torch.nn.Linear(DEFAULT_OBS_SIZE + DEFAULT_ACTION_SIZE, 1)

    def forward(self, observations, actions):
        """ Maps (observations, action) into values """
        x = torch.cat([observations, actions])

        return self.layer(x)


class MockedPolicyModel(torch.nn.Module):
    """ Mocked PolicyModel """

    def __init__(self):
        super().__init__()

        self.mean = torch.nn.Linear(DEFAULT_OBS_SIZE, DEFAULT_ACTION_SIZE)
        self.log_std = torch.nn.Linear(DEFAULT_OBS_SIZE, DEFAULT_ACTION_SIZE)

    def forward(self, observations):
        """ Maps observations into actions """
        mean = self.mean(observations)
        log_std = self.log_std(observations)

        std = log_std.exp()

        return mean, std


def get_mocked_q_model() -> torch.nn.Module:
    """ Returns a Qmodel with presaved weights if it does not exists """
    model_path = PATH + 'MockedQModel.pt'
    model = MockedQmodel()

    try:
        model.load_state_dict(torch.load(model_path))

    except FileNotFoundError:
        torch.save(model.state_dict(), model_path)

    return model


def get_mocked_policy_model() -> torch.nn.Module:
    """ Returns a PolicyModel with presaved weights if it does not exists """
    model_path = PATH + 'MockedPolicyModel.pt'
    model = MockedPolicyModel()

    try:
        model.load_state_dict(torch.load(model_path))

    except FileNotFoundError:
        torch.save(model.state_dict(), model_path)

    return model
