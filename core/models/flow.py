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
        timestep: int = 0,
    ) -> None:
        self.name: str = name
        self.sources: List[Union[Facility, ControlledFacility]] = sources
        self.destination: Union[Facility, ControlledFacility] = destination
        self.max_capacity: float = max_capacity
        self.timestep = timestep

    def determine_source_outflow(self) -> float:
        return sum(source.outflow for source in self.sources)

    def set_destination_inflow(self) -> None:
        self.destination.inflow = self.determine_source_outflow()

    def determine_info(self) -> dict:
        return {"flow": self.determine_source_outflow()}

    def step(self) -> Tuple[Optional[ObsType], float, bool, bool, dict]:
        self.set_destination_inflow()

        terminated = self.determine_source_outflow() > self.max_capacity
        truncated = False
        reward = float("-inf") if terminated else 0.0
        info = self.determine_info()

        self.timestep += 1

        return None, reward, terminated, truncated, info


class Inflow(Flow):
    def __init__(
        self,
        name: str,
        destination: Union[Facility, ControlledFacility],
        max_capacity: float,
        inflow: List[float],
        timestep: int = 0,
    ) -> None:
        super().__init__(name, None, destination, max_capacity, timestep)
        self.inflow: float = inflow

    def determine_source_outflow(self) -> float:
        return self.inflow[self.timestep]


class Outflow(Flow):
    def __init__(
        self, name: str, sources: List[Union[Facility, ControlledFacility]], max_capacity: float, timestep: int = 0
    ) -> None:
        super().__init__(name, sources, None, max_capacity, timestep)

    def set_destination_inflow(self) -> None:
        pass
