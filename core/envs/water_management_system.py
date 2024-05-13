import gymnasium as gym
from gymnasium.spaces import Space, Dict as Dct
from gymnasium.core import ObsType, RenderFrame
from typing import Any, List, Union, SupportsFloat, Optional, Tuple, Dict
from core.models.flow import Flow
from core.models.facility import Facility, ControlledFacility


class WaterManagementSystem(gym.Env):
    def __init__(
        self,
        water_systems: List[Union[Facility, ControlledFacility, Flow]],
        rewards: dict,
        seed=42,
    ) -> None:
        self.water_systems: List[Union[Facility, ControlledFacility, Flow]] = (
            water_systems
        )
        self.rewards = rewards
        self.seed: int = seed

        self.observation_space: Space = self._determine_observation_space()
        self.action_space: Space = self._determine_action_space()

    def _determine_observation_space(self) -> Dct:
        return Dct(
            {
                water_system.name: water_system.observation_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            },
            self.seed,
        )

    def _determine_action_space(self) -> Dct:
        return Dct(
            {
                water_system.name: water_system.action_space
                for water_system in self.water_systems
                if isinstance(water_system, ControlledFacility)
            },
            self.seed,
        )

    def _determine_info(self) -> Dict[str, Any]:
        # TODO: decide on what we wnat to output in the info.
        return {"water_systems": self.water_systems}

    def reset(
        self, seed: Optional[int] = None, options: Optional[dict] = None
    ) -> Tuple[ObsType, Dict[str, Any]]:
        # We need the following line to seed self.np_random.
        super().reset(seed=seed)

        # TODO: Figure out the reset in facility
        # for water_system in self.water_systems:
        #     water_system.reset()

        return self._determine_observation_space(), self._determine_info()

    def step(self, action: Dct) -> Tuple[ObsType, list, bool, bool, dict]:
        final_observation = {}
        final_reward = self.rewards
        final_terminated = False
        final_truncated = False
        final_info = {}

        for water_system in self.water_systems:
            if isinstance(water_system, ControlledFacility):
                observation, reward, terminated, truncated, info = water_system.step(
                    action[water_system.name]
                )
            elif isinstance(water_system, Facility) or isinstance(water_system, Flow):
                observation, reward, terminated, truncated, info = water_system.step()
            else:
                raise ValueError()

            # Set observation for a specific Facility.
            final_observation[water_system.name] = observation
            # Add reward to the objective assigned to this Facility (unless it is a Flow).
            if isinstance(water_system, Facility) or isinstance(
                water_system, ControlledFacility
            ):
                final_reward[water_system.objective_name] += reward
            # Determine whether program should stop
            final_terminated = final_terminated or terminated
            final_truncated = final_truncated or truncated
            # Store additional information
            final_info[water_system.name] = info

        return (
            list(final_observation.values()),
            list(final_reward.values()),
            final_terminated,
            final_truncated,
            final_info,
        )

    def close(self) -> None:
        # TODO: implement if needed, e.g. for closing opened rendering frames.
        pass

    def render(self) -> Union[RenderFrame, List[RenderFrame], None]:
        # TODO: implement if needed, for rendering simulation.
        pass
