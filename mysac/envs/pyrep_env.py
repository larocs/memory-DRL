import math
from typing import List

import numpy as np
from gym import Env, spaces
from pyrep import PyRep
from pyrep.backend import sim
from pyrep.backend.sim import (simGetJointMatrix, simGetObjectVelocity,
                               simSetSphericalJointMatrix)
from pyrep.objects.joint import Joint
from pyrep.objects.shape import Shape


def inverse_logit(x):
    return x
    return 1 / (1 + math.exp(-x))


def min_max_norm(x, min, max):
    return (x - min)/(max - min)


class CartPoleEnv(Env):
    SCENES_FOLDER = ('/home/samuel/Develop/IC/my_own_sac/mysac/envs/'
                     'coppelia_scenes/')

    SCENE_FILE = 'cart_pole_2d_up.ttt'

    def __init__(self,
                 last_n_frames=50,
                 buffer_for_rnn=False,
                 normalize_observation=False,
                 headless=False,
                 height_limit=-0.4,
                 max_steps_under_height_limit=250,
                 random_init: bool = True):
        self.total_steps = 0
        self.total_reward = 0

        self.height_limit = -0.4
        self.max_steps_under_height_limit = max_steps_under_height_limit
        self.total_steps_under_height_limit = 0
        self.random_init = random_init

        self.pr = PyRep()
        self.pr.launch(self.SCENES_FOLDER + self.SCENE_FILE, headless=headless)
        self.pr.start()

        self.bearing = Joint(name_or_handle='bearing')
        self.slider = Joint(name_or_handle='slider')
        self.slider2 = Joint(name_or_handle='slider2')
        self.cart = Shape(name_or_handle='cart')
        self.mass = Shape(name_or_handle='mass')

        self.episode_steps = 0
        self.last_n_frames = last_n_frames
        self.normalize_observation = normalize_observation

        if not buffer_for_rnn:
            obs = np.array([np.inf] * 10)

        else:
            # (Sequence len, obs dim)
            obs = np.inf * np.ones((10,))

        act = np.array([2.]*2)
        self.action_space = spaces.Box(-act, act)
        self.observation_space = spaces.Box(-obs, obs)

        self.buffer_for_rnn = buffer_for_rnn
        if self.buffer_for_rnn:
            # Sequence len: 50, obs len: 10
            self.last_actions = np.zeros((last_n_frames, 10)).astype('float32')

    def render(self, _):
        pass

    def random_init_state(self):
        if not self.random_init:
            return

        simSetSphericalJointMatrix(
            self.bearing._handle,
            np.random.uniform(-1, 1, 12).tolist()
        )

        self.slider.set_joint_position(np.random.uniform(-1, 1))
        self.slider2.set_joint_position(np.random.uniform(-1, 1))

    def observe(self):
        cartpos = self.cart.get_position()
        masspos = self.mass.get_position()
        cartvel, _ = simGetObjectVelocity(self.cart.get_handle())
        massvel, _ = simGetObjectVelocity(self.mass.get_handle())

        if not self.normalize_observation:
            obs = np.array([
                cartpos[0], cartpos[1],
                cartvel[0], cartvel[1],
                masspos[0], masspos[1], masspos[2],
                massvel[0], massvel[1], massvel[2]
            ]).astype('float32')

        else:
            obs = [min_max_norm(cartpos[0], -0.55, 0.55),
                   min_max_norm(cartpos[1], -0.55, 0.55),
                   min_max_norm(cartvel[0], -8, 8),
                   min_max_norm(cartvel[1], -8, 8),
                   min_max_norm(masspos[0], -1, 1),
                   min_max_norm(masspos[1], -1, 1),
                   min_max_norm(masspos[2], -1, 1),
                   min_max_norm(massvel[0], -25, 25),
                   min_max_norm(massvel[1], -25, 25),
                   min_max_norm(massvel[2], -25, 25)]

            obs = np.array(obs).astype('float32')

        if self.buffer_for_rnn:
            self.last_actions = np.roll(self.last_actions, -10)
            self.last_actions[-1] = obs
            obs = self.last_actions

        return obs.astype('float32')

    def step(self, action):
        self.total_steps += 1

        v, v1 = 2*action

        self.slider.set_joint_target_velocity(v)
        self.slider2.set_joint_target_velocity(v1)
        self.pr.step()

        obs = self.observe()
        height_of_mass = self.mass.get_position()[2]
        reward = height_of_mass - (v**2) * 0.0005 - (v1**2) * 0.0005

        # Controls the end of the episode by the height limit
        if height_of_mass < self.height_limit:
            self.total_steps_under_height_limit += 1
        else:
            self.total_steps_under_height_limit = 0

        # if self.total_steps_under_height_limit >= self.max_steps_under_height_limit:
        #     done = True

        self.episode_steps += 1
        self.total_reward += reward

        return obs, reward, False, ''

    def reset(self):
        self.total_reward = 0
        self.episode_steps = 0

        self.total_steps_under_height_limit = 0

        if self.buffer_for_rnn:
            # Sequence len: 100, obs len: 10
            self.last_actions = np.zeros((self.last_n_frames, 10))

        self.pr.stop()
        self.random_init_state()
        self.pr.start()

        return self.observe()


if __name__ == '__main__':
    import pandas as pd

    env = CartPoleEnv(normalize_observation=True)

    print('Initing')

    states = []
    for e in range(100):
        print("Episode ", e)
        env.reset()

        for ts in range(2):
            # print("Step ", ts)
            state, _, _, _ = env.step(env.action_space.sample())

            states.append(state)

    df = pd.DataFrame(states)
    df.to_csv('stat_hist.csv')
