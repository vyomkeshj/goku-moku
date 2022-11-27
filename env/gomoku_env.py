import numpy as np


import gym
from gym import spaces
from math import *
import torch

class fast_environment(gym.Env):

    def __init__(self):
        self.action_count = 0;
        self._max_episode_steps = 100

        self.action_space = spaces.Box(low=np.array([-4.0000, -4.0000, -4.0000, -4.0000, -4.0000, -4.0000]),
                                       high=np.array([4.0000, 4.0000, 4.0000, 4.0000, 4.0000, 4.0000]),
                                       dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.array([-pi, -pi, -pi, -pi, -pi, -pi, -1, -1, -1, -pi, -1, -1, -1, -1, -1, -1,
                          -pi, -pi, -pi, -pi, -pi, -pi, -1, -1, -1, -pi, -1, -1, -1, -1, -1, -1,
                          -pi, -pi, -pi, -pi, -pi, -pi, -1, -1, -1, -pi, -1, -1, -1, -1, -1, -1,
                          -pi, -pi, -pi, -pi, -pi, -pi, -1, -1, -1, -pi, -1, -1, -1, -1, -1, -1, ],
                         dtype=np.float32),

            high=np.array([pi, pi, pi, pi, pi, pi, 1, 1, 1, pi, 1, 1, 1, 1, 1, 1,
                           pi, pi, pi, pi, pi, pi, 1, 1, 1, pi, 1, 1, 1, 1, 1, 1,
                           pi, pi, pi, pi, pi, pi, 1, 1, 1, pi, 1, 1, 1, 1, 1, 1,
                           pi, pi, pi, pi, pi, pi, 1, 1, 1, pi, 1, 1, 1, 1, 1, 1, ]),
            dtype=np.float32)

        self.num_active_joints = self.action_space.shape[0]  # number of joints to be updated (from base to wrist)
        self.robot_mdp = robot_mdp(self.num_active_joints)

        self.observation_not_angles_size = 10  # observation that is not joint angles (vector distance, etc)
        self.observation_buffer_size = 4  # history size
        self.single_observation_size = (self.robot_mdp.dof + self.observation_not_angles_size)

        self.observation_size = (self.robot_mdp.dof + self.observation_not_angles_size) * self.observation_buffer_size  # this is uses elsewhere

        self.observation_buffer = torch.zeros(self.observation_size)

    def get_net_observation(self, current_observation, flush):
        local_buffer = torch.cat(
            (self.observation_buffer[self.single_observation_size: self.observation_size], current_observation))
        self.observation_buffer = local_buffer
        if flush:
            self.observation_buffer = torch.zeros(self.observation_size)

        return local_buffer

    def step(self, action):
        # depending on the action size, append zeros to it for rest of the joint to signify static joints using np.pad
        num_zeros_to_append = self.robot_mdp.dof - self.num_active_joints
        action = np.pad(action, (0, num_zeros_to_append), 'constant')

        # print("action done = ",action)
        observation = self.robot_mdp.update_angle(action)
        # return <angles and vec difference>, <reward>, <done_status>. The first n_dof elements are angles, next observation_not_angles_size are vec difference
        observed_angles = observation[0:self.num_active_joints]
        observed_difference = observation[self.robot_mdp.dof: (self.robot_mdp.dof + self.observation_not_angles_size)]
        net_obs = torch.cat((observed_angles, observed_difference))
        # print("net observation = ",net_obs)
        reward_received = observation[self.single_observation_size]
        done_status = observation[self.single_observation_size + 1]
        # print("action = ", action, "observation = ", net_obs, "done =", done_status)
        # print("stepping ", action)
        return self.get_net_observation(net_obs, False), reward_received, done_status, {}

    def reset(self):
        observation = self.robot_mdp.reset_robot()

        observed_angles = observation[0:self.num_active_joints]

        observed_difference = observation[self.robot_mdp.dof: (self.robot_mdp.dof + self.observation_not_angles_size)]
        net_obs = torch.cat((observed_angles, observed_difference))

        return self.get_net_observation(net_obs, True)  # on reset, the observation is only the state of the environment

    def render(self, mode='human'):
        pass
