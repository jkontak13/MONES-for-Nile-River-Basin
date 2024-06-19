import numpy as np
import gymnasium as gym
from gymnasium.spaces.dict import Dict
from core.envs.water_management_system import WaterManagementSystem


class ReshapeArrayAction(gym.ActionWrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: WaterManagementSystem):
        assert isinstance(env.action_space, Dict)

        self.slices = {}
        current_index = 0

        for name, sub_action_space in env.action_space.items():
            self.slices[name] = slice(current_index, current_index + np.prod(sub_action_space.shape))
            current_index += np.prod(sub_action_space.shape)

        gym.utils.RecordConstructorArgs.__init__(self)
        gym.ActionWrapper.__init__(self, env)

    def action(self, action):
        reshaped_actions = {}

        for name, sub_action_space in self.env.action_space.items():
            reshaped_actions[name] = np.reshape(action[self.slices[name]], sub_action_space.shape)

        return reshaped_actions
