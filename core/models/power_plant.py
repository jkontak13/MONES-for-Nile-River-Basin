from core.models.facility import Facility

class PowerPlant(Facility):

    def __init__(self, inflow: float, outflow: float, max_capacity: float, power_production: float, water_usage: float) -> None:
        super().__init__(inflow, outflow, max_capacity)
        self.power_production = power_production
        self.water_usage = water_usage
    
    def determine_reward(self) -> float:
        return self.inflow * self.power_production

    def determine_consumption(self) -> float:
        return self.inflow * self.water_usage
