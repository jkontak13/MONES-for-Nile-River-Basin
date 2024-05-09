import numpy as np
import gymnasium as gym
from gymnasium.spaces import Space, Dict
from gymnasium.core import ObsType
from typing import Any, List, Union, Optional, SupportsFloat
from core.models.flow import Flow
from core.models.facility import Facility, ControlledFacility


class WaterManagementSystem(gym.Env):
    def __init__(self, water_systems: List[Union[Facility, ControlledFacility, Flow]], seed=42) -> None:
        self.water_systems: List[Union[Facility, ControlledFacility, Flow]] = water_systems
        self.seed: int = seed

        self.observation_space: Space = self._determine_observation_space()
        self.action_space: Space = self._determine_action_space()

    def _determine_observation_space(self) -> Dict:
        return Dict(
            {
                water_system.id: water_system.observation_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            },
            self.seed,
        )

    def _determine_action_space(self) -> Dict:
        return Dict(
            {
                water_system.id: water_system.action_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            },
            self.seed,
        )

    def _determine_info(self) -> dict[str, Any]:
        # TODO: decide on what we wnat to output in the info.
        return {"water_systems": self.water_systems}

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> tuple[ObsType, dict[str, Any]]:
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        # TODO: Figure out the reset in facility
        # for water_system in self.water_systems:
        #     water_system.reset()

        return self._determine_observation_space(), self._determine_info()

    def step(self, action: Dict) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:

        final_observation = {}
        final_reward = 0
        final_terminated = False
        final_truncated = False
        final_info = {}

        for water_systems in self.water_systems:
            if isinstance(water_systems, ControlledFacility):
                observation, reward, terminated, truncated, info = water_systems.step(action[water_systems.id])
            elif isinstance(water_systems, Facility) or isinstance(water_systems, Flow):
                observation, reward, terminated, truncated, info = water_systems.step()
            else:
                raise ValueError()

            final_observation[water_systems.id] = observation
            final_reward += reward
            final_terminated = final_terminated or terminated
            final_truncated = final_truncated or truncated
            final_info[water_systems.id] = info

        # TODO: Flatten final_observation and create numpy array out of it
        # TODO: Make the final_reward an array instead of a single number
        return final_observation, final_reward, final_terminated, final_truncated, final_info

    def close(self) -> None:
        # TODO: implement if needed, e.g. for closing opened rendering frames.
        pass
