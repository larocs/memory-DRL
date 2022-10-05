from typing import List, Tuple

import numpy as np
from gym import spaces
from mysac.envs.nao.nao_env import vectorized_to_interval
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape
from pyrep.robots.humanoids.marta_robot import MartaRobot

MAX_VELOCITY = 3 * 50e-3
ENERGY_COST_THRESHOLD = 4


class MartaWalkEnv:
    """
    Environment for teaching Marta how to forward walking, similar to NaoEnv
    """

    SCENES_FOLDER = ('./mysac/envs/coppelia_scenes/')

    SCENE_FILE = 'Martabot2.ttt'

    def __init__(
            self,
            headless: bool = True,
            random_initialization: bool = True,
            max_velocity: float = MAX_VELOCITY,
            energy_cost_threshold: float = ENERGY_COST_THRESHOLD,
            * args,
            **kwargs
    ):
        self.pr = PyRep()
        self.pr.launch(self.SCENES_FOLDER + self.SCENE_FILE, headless=headless)
        self.pr.start()

        self.robot = MartaRobot()

        self.delayed_energy_cost = []
        self.past_joint_velocities = []
        self.use_delayed_energy_cost = True

        self.load_shapes()
        self.total_steps = 0

        self.all_joints = [
            Joint(handle)
            for handle in self.robot.joint_handles
        ]

        limits = zip(self.robot.low_act_limits, self.robot.high_act_limits)
        self.joint_limits = np.array(
            [(low, high) for low, high in limits]
        )

        self.action_space = spaces.Box(
            low=np.array(23 * [-1]),
            high=np.array(23 * [1]),
        )

        self.max_velocity = max_velocity
        self.energy_cost_threshold = energy_cost_threshold

    def load_shapes(self) -> None:
        """
        Load some of Marta shapes for retrieving states
        """
        self.head = Shape(name_or_handle=self.robot.shapes_handle[0])
        self.chest = Shape(name_or_handle=self.robot.shapes_handle[1])

        self.l_foot = Shape(
            name_or_handle='Marta_leftfoot_link2_respondable'
        )

        self.r_foot = Shape(
            name_or_handle='Marta_rightfoot_link2_respondable'
        )

    def get_foot_sensor_signal(self) -> List[float]:
        """
        Returns the signal for all the foot force sensors
        """
        readings = []

        for force_sensor in self.robot.force_sensors_handle:
            _, force, torque = sim.simReadForceSensor(force_sensor)

            readings.extend(force)
            readings.extend(torque)

        return readings

    def get_observation(self) -> np.array:
        """
        Returns an Numpy Array with the env observations
        """
        _, _, head_z = self.head.get_position()

        orientation_x, orientation_y, orientation_z = \
            self.chest.get_orientation()

        linear, angular = sim.simGetObjectVelocity(
            objectHandle=self.chest._handle)

        joint_positions = [
            joint.get_joint_position() for joint in self.all_joints
        ]

        return np.array(
            (
                head_z,
                orientation_x,
                orientation_y,
                orientation_z,
                *linear,
                *angular,
                *joint_positions,
                *self.get_foot_sensor_signal()
            ),
            dtype='float32'
        )

    def get_reward(self) -> float:
        """
        Returns the reward signal in the current simulation state
        """
        last_position = getattr(
            self,
            'last_position',
            (-2.32, -1.19)
        )

        chest_handle = self.robot.shapes_handle[1]
        x, y, _ = sim.simGetObjectPosition(chest_handle, -1)

        max_velocity = self.max_velocity
        delta_x = np.clip(x - last_position[0], -max_velocity, max_velocity)
        delta_y = np.clip(y - last_position[1], -max_velocity, max_velocity)

        delta_x *= 500 if delta_x > 0 else 1000
        delta_y = -250 * np.abs(delta_y)

        reward = delta_x + delta_y

        x_orientation, _, z_orientation = sim.simGetObjectOrientation(
            chest_handle, -1)

        if x_orientation < -0.4 or x_orientation > 0.4:
            reward *= 0.8

        if z_orientation < -0.4 or z_orientation > 0.4:
            reward *= 0.8

        if self.foots_collision():
            reward *= 0.8

        self.last_position = (x, y)

        return reward

    def foots_collision(self) -> bool:
        """
        Returns true if the NAO foot respondable parts collide
        """
        return self.l_foot.check_collision(self.r_foot)

    def reset(self, random_initialization: bool = True) -> np.array:
        """
        Reset the env to the initial state, optionally doing a random action

        Args:
            random_initialization: if True, will execute a random action from
                the action_space attribute
        """
        self.position_history = []
        self.past_joint_velocities = []
        self.delayed_energy_cost = []
        self.random_initialization = random_initialization

        self.pr.stop()
        self.pr.start()

        if self.random_initialization:
            self.step(action=self.action_space.sample())

        self.total_steps = 0

        return self.get_observation()

    def step(self, action: np.array) -> Tuple[np.array, float, bool, str]:
        """
        Executes an action in the environment

        Args:
            action: an array of shape action_space.shape

        Returns:
            A tuple containing the current state, the reward signal, the done
            signal and an optional information string
        """
        self.total_steps += 1

        x, y, z = self.head.get_position()

        done = z < 0.35
        done |= 2.15 < y < -2.15
        done |= -2.32 > x > 2.5

        action = vectorized_to_interval(
            limits=self.joint_limits,
            actions=action
        )

        for target_position, joint in zip(action, self.all_joints):
            joint.set_joint_target_position(target_position)

        self.pr.step()

        self.compute_delayed_energy_cost()

        # We use this for energy cost
        self.past_joint_velocities = [
            j.get_joint_velocity() for j in self.all_joints
        ]

        return self.get_observation(), self.get_reward(), done, ''

    def compute_delayed_energy_cost(self) -> float:
        """
        Returns the energy cost of the >last< action
        """
        if not self.past_joint_velocities:
            return

        # The current velocity is due to the last action
        current_velocity = np.array(
            [j.get_joint_velocity() for j in self.all_joints]
        )

        past_joint_velocities = np.array(
            self.past_joint_velocities
        )

        cost = np.abs(current_velocity - past_joint_velocities).mean()

        self.delayed_energy_cost.append(cost)

    def post_episode_sampling_callback(
            self, rewards: List[np.array]) -> List[np.array]:
        """
        Called by the sampler if the reward must be calculated with a delayed
        effect

        Args:
            rewards: the rewards originally collected for the episode
        """
        if not self.use_delayed_energy_cost:
            return rewards

        self.compute_delayed_energy_cost()

        delayed_energy_cost = self.delayed_energy_cost
        if self.random_initialization:
            delayed_energy_cost = delayed_energy_cost[1:]

        new_rewards = []
        for reward, energy_cost in zip(rewards, self.delayed_energy_cost):
            if energy_cost > self.energy_cost_threshold:
                new_rewards.append(0.8 * reward)

            else:
                new_rewards.append(reward)

        return new_rewards


class RecurrentMartaWalkEnv(MartaWalkEnv):
    """
    Adaptor for MartaWalkEnv env that handles the observations as sequence of
    frames

    Args:
        frames: the number of frames to keep
    """

    def __init__(self, frames: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.frames = frames

        self.last_observations = np.zeros(
            (self.frames, 81)
        ).astype('float32')

    def _append_frame(self, frame: np.array) -> np.array:
        """
        Append a frame to the buffers end
        """
        self.last_observations = np.roll(self.last_observations, -81)
        self.last_observations[-1] = frame

        return self.last_observations

    def reset(self, *args, **kwargs) -> np.array:
        """
        Reset the environment and the observation buffer
        """
        self.last_observations = np.zeros(
            (self.frames, 81)
        ).astype('float32')

        return super().reset(*args, **kwargs)

    def get_observation(self):
        """
        Returns the observation as done by the base class, but in the form of
        a buffer
        """
        frame = super().get_observation()

        return self._append_frame(frame=frame)
