from abc import ABC, abstractmethod
import numpy as np
from gymnasium.spaces import Space
from gymnasium.core import ObsType, ActType
from typing import SupportsFloat, Tuple
from core.models.objective import Objective


class Facility(ABC):
    def __init__(self, name: str, objective_function=Objective.no_objective, objective_name: str = "") -> None:
        self.name: str = name
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

        self.objective_function = objective_function
        self.objective_name = objective_name

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_consumption(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_info(self) -> dict:
        raise NotImplementedError()

    def is_terminated(self) -> bool:
        return False

    def is_truncated(self) -> bool:
        return False

    def get_outflow_at_timestep(self, timestep: int) -> float:
        if timestep < 0:
            return self.all_outflow[0]
        else:
            return self.all_outflow[timestep]

    def set_inflow(self, timestep: int, inflow: float) -> None:
        if len(self.all_inflow) != timestep:
            raise IndexError

        self.all_inflow.append(inflow)

    def step(self) -> Tuple[ObsType, float, bool, bool, dict]:
        self.all_outflow.append(self.all_inflow[self.timestep] - self.determine_consumption())
        # TODO: Determine if we need to satisy any terminating codnitions for facility.
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info

    def reset(self) -> None:
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []


class ControlledFacility(ABC):
    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: ActType,
        objective_function=Objective.no_objective,
        objective_name: str = "",
        max_capacity: float = float("Inf"),
    ) -> None:
        self.name: str = name
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []

        self.observation_space: Space = observation_space
        self.action_space: Space = action_space

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.max_capacity: float = max_capacity

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_outflow(self, action: ActType) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_info(self) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def determine_observation(self) -> ObsType:
        raise NotImplementedError()

    @abstractmethod
    def is_terminated(self) -> bool:
        raise NotImplementedError()

    def is_truncated(self) -> bool:
        return False

    def get_outflow_at_timestep(self, timestep: int) -> float:
        if timestep < 0:
            return self.all_outflow[0]
        else:
            return self.all_outflow[timestep]

    def set_inflow(self, timestep: int, inflow: float) -> None:
        if len(self.all_inflow) != timestep:
            raise IndexError

        self.all_inflow.append(inflow)

    def step(self, action: ActType) -> Tuple[ObsType, SupportsFloat, bool, bool, dict]:
        self.all_outflow.append(self.determine_outflow(action))
        # TODO: Change stored_water to multiple outflows.

        observation = self.determine_observation()
        reward = self.determine_reward()
        terminated = self.is_terminated()
        truncated = self.is_truncated()
        info = self.determine_info()

        self.timestep += 1

        return (
            observation,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self) -> None:
        self.timestep: int = 0
        self.all_inflow: list[float] = []
        self.all_outflow: list[float] = []
