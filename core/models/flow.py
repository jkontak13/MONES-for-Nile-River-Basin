from core.models.facility import Facility, ControlledFacility
from typing import Union

class Flow:
    def __init__(self, source: Union[Facility, ControlledFacility], destination: Union[Facility, ControlledFacility], max_capacity: float)  -> None:
        self.source = source
        self.destination = destination
        self.max_capacity = max_capacity

    def step(self) -> None:
        self.destination.inflow = self.source.outflow

        if self.source.outflow > self.max_capacity:
            return float('-inf')
        else:
            return 0
        
class Inflow(Flow):
    def __init__(self, destination: Union[Facility, ControlledFacility], inflow: float)  -> None:
        self.destination = destination
        self.inflow = inflow

    def step(self) -> None:
        self.destination.inflow = self.inflow
        
        return 0
        
class Outflow(Flow):
    def __init__(self, source: Union[Facility, ControlledFacility], max_capacity: float)  -> None:
        self.source = source
        self.max_capacity = max_capacity

    def step(self) -> None:        
        if self.source.outflow > self.max_capacity:
            return float('-inf')
        else:
            return 0
