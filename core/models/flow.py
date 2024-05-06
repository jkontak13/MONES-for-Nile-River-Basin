from core.models.facility import Facility, ControlledFacility
from typing import Optional, Union
from gymnasium.core import ObsType
from typing import SupportsFloat


class Flow:
    def __init__(
        self,
        id: str,
        source: Optional[Union[Facility, ControlledFacility]],
        destination: Optional[Union[Facility, ControlledFacility]],
        max_capacity: float,
    ) -> None:
        self.id: str = id
        self.source: Union[Facility, ControlledFacility] = source
        self.destination: Union[Facility, ControlledFacility] = destination
        self.max_capacity: float = max_capacity

    def determine_source_outflow(self) -> float:
        return self.source.outflow

    def set_destination_inflow(self) -> None:
        self.destination.inflow = self.determine_source_outflow()

    def determine_info(self) -> str:
        # TODO: Determine info for Flow.
        return ""

    def step(self) -> tuple[ObsType, float, bool, bool, dict]:
        self.set_destination_inflow()

        terminated = self.determine_source_outflow() > self.max_capacity
        reward = float("-inf") if terminated else 0

        return None, reward, terminated, False, self.determine_info()


class Inflow(Flow):
    def __init__(
        self, id: str, destination: Union[Facility, ControlledFacility], max_capacity: float, inflow: float
    ) -> None:
        super().__init__(id, None, destination, max_capacity)
        self.inflow: float = inflow

    def determine_source_outflow(self) -> float:
        return self.inflow


class Outflow(Flow):
    def __init__(self, id: str, source: Union[Facility, ControlledFacility], max_capacity: float) -> None:
        super().__init__(id, source, None, max_capacity)

    def set_destination_inflow(self) -> None:
        pass
