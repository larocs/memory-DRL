from mysac.envs.nao import WalkingNao
import numpy as np

OBS_SIZE = 30


class RecurrentNAO(WalkingNao):
    """
    Adaptor for WalkingNao env that handles the observations as sequence of
    frames

    Args:
        frames: the number of frames to keep
    """

    def __init__(self, frames: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frames = frames

        self.last_observations = np.zeros(
            (self.frames, OBS_SIZE)
        ).astype('float32')

    def _append_frame(self, frame: np.array) -> np.array:
        """
        Append a frame to the buffers end
        """
        self.last_observations = np.roll(self.last_observations, -OBS_SIZE)
        self.last_observations[-1] = frame

        return self.last_observations

    def reset(self, *args, **kwargs) -> np.array:
        """
        Reset the environment and the observation buffer
        """
        self.last_observations = np.zeros(
            (self.frames, OBS_SIZE)
        ).astype('float32')

        return super().reset(*args, **kwargs)

    def get_observation(self):
        """
        Returns the observation as done by the base class, but in the form of
        a buffer
        """
        frame = super().get_observation()

        return self._append_frame(frame=frame)
