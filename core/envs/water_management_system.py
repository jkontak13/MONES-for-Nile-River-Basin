import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import Optional, ActType, ObsType
from typing import List, SupportsFloat
from core.models.flow import Flow
from core.models.facility import Facility


class WaterManagementSystem(gym.Env):
    def __init__(self, facilities: List[Facility], flows: List[Flow]) -> None:
        # TODO: decide on the type of observation space.
        self.observation_space = gym.spaces.Box(low=0, high=100, shape=(1,))
        
        # TODO: decide on the type of action space.
        self.action_space = gym.spaces.Discrete(2)
        
        # TODO: decide on the type of reward space.
        self.reward_space = gym.spaces.Box(-np.inf, np.inf, shape=(1,))

    def _get_obs(self):
        # TODO: decide on how we represent observation.
        return {"agent": self._agent_location, "target": self._target_location}
    
    def _get_info(self):
        # TODO: decide on what we wnat to output in the info.
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
    
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict]:
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)
        self.time_step = 0

        # TODO: Randomly initialize state of the enviroment.

        if self.initial_state is not None:
            state = self.initial_state
        else:
            if not self.penalize:
                state = self.np_random.choice(WaterManagementSystem.s_init, size=1)
            else:
                state = self.np_random.integers(0, 160, size=1)

        self.state = np.array(state, dtype=np.float32)
        
        # observation = self._get_obs()
        # info = self._get_info()

        # return observation, info

        if self.render_mode == "human":
            self.render()

        return self.state, {}
    
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )
        # An episode is done iff the agent has reached the target
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info
    
    def close(self) -> None:
        # TODO: implement if needed, e.g. for closing opened rendering frames.
        pass
