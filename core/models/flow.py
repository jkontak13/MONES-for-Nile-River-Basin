from typing import Optional, List, Union, Tuple
from core.models.facility import Facility, ControlledFacility
from gymnasium.core import ObsType


class Flow:
    def __init__(
        self,
        name: str,
        sources: List[Union[Facility, ControlledFacility]],
        destination: Optional[Union[Facility, ControlledFacility]],
        max_capacity: float,
        evaporation_rate: float = 0.0,
    ) -> None:
        self.name: str = name
        self.sources: List[Union[Facility, ControlledFacility]] = sources
        self.destination: Union[Facility, ControlledFacility] = destination
        self.max_capacity: float = max_capacity
        self.evaporation_rate = evaporation_rate
        self.timestep = 0

    def determine_source_outflow(self) -> float:
        return sum(source.outflow for source in self.sources)

    def set_destination_inflow(self) -> None:
        self.destination.inflow = self.determine_source_outflow() * (1.0 - self.evaporation_rate)

    def is_truncated(self) -> bool:
        return False

    def determine_info(self) -> dict:
        return {"name": self.name, "flow": self.determine_source_outflow()}

    def step(self) -> Tuple[Optional[ObsType], float, bool, bool, dict]:
        self.set_destination_inflow()

        terminated = self.determine_source_outflow() > self.max_capacity
        truncated = self.is_truncated()
        reward = float("-inf") if terminated else 0.0
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info

    def reset(self) -> None:
        self.timestep = 0


class Inflow(Flow):
    def __init__(
        self,
        name: str,
        destination: Union[Facility, ControlledFacility],
        max_capacity: float,
        all_inflow: List[float],
        evaporation_rate: float = 0.0,
    ) -> None:
        super().__init__(name, None, destination, max_capacity, evaporation_rate)
        self.all_inflow: List[float] = all_inflow

    def determine_source_outflow(self) -> float:
        return self.all_inflow[self.timestep % len(self.all_inflow)]

    def is_truncated(self) -> bool:
        return self.timestep >= len(self.all_inflow)


class Outflow(Flow):
    def __init__(
        self,
        name: str,
        sources: List[Union[Facility, ControlledFacility]],
        max_capacity: float,
    ) -> None:
        super().__init__(name, sources, None, max_capacity, evaporation_rate=0.0)

    def set_destination_inflow(self) -> None:
        pass
