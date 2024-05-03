from abc import ABC, abstractmethod

class Facility(ABC):
    def __init__(self, inflow: float, outflow: float) -> None:
        self.inflow = inflow
        self.outflow = outflow

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def determine_consumption(self) -> float:
        raise NotImplementedError()
    
    def step(self) -> float:
        self.outflow = self.inflow - self.determine_consumption()

        return self.determine_reward()
        
class ControlledFacility(ABC):
    def __init__(self, inflow: float, outflow: float, max_capacity: float, stored_water: float) -> None:
        self.inflow = inflow
        self.outflow = outflow
        self.max_capacity = max_capacity
        self.stored_water = stored_water

    @abstractmethod
    def determine_reward(self) -> float:
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: float) -> float:
        raise NotImplementedError()
    