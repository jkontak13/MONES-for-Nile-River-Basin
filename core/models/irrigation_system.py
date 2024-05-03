from core.models.facility import Facility

class IrrigationSystem(Facility):

    def __init__(self, inflow: float, outflow: float, max_capacity: float, water_usage: float) -> None:
        super().__init__(inflow, outflow, max_capacity)
        self.water_usage = water_usage
    
    def determine_reward(self) -> float:
        return (self.outflow - self.inflow) / self.water_usage

    def determine_consumption(self) -> float:
        return min(self.water_usage, self.inflow)
