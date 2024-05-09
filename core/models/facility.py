from abc import ABC, abstractmethod
from gymnasium.spaces import Space, Box
from gymnasium.core import ObsType, ActType
from typing import SupportsFloat, Tuple


class Facility(ABC):
    def __init__(self, id: str) -> None:
        self.id: str = id
        self.inflow: float = 0
        self.outflow: float = 0

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_consumption(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_info(self) -> dict:
        raise NotImplementedError()

    def step(self) -> tuple[ObsType, float, bool, bool, dict]:
        self.outflow = self.inflow - self.determine_consumption()
        # TODO: Determine if we need to satisy any terminating codnitions for facility.
        terminated = False

        return None, self.determine_reward(), terminated, False, self.determine_info()


class ControlledFacility(ABC):
    def __init__(self, id: str, observation_space: Space, action_space: ActType, max_capacity: float = float("Inf")) -> None:
        self.id: str = id
        self.inflow: float = 0
        self.outflow: float = 0

        self.observation_space: Space = observation_space
        self.action_space: Space = action_space

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

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict]:
        self.outflow = self.determine_outflow(action)
        # TODO: Change stored_water to multiple outflows.

        return (
            self.determine_observation(),
            self.determine_reward(),
            self.is_terminated(),
            False,
            self.determine_info(),
        )
