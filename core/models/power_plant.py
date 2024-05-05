from core.models.facility import Facility


class PowerPlant(Facility):

    def __init__(self, id: str, power_production: float, water_usage: float) -> None:
        super().__init__(id)
        self.power_production: float = power_production
        self.water_usage: float = water_usage

    def determine_reward(self) -> float:
        return self.inflow * self.power_production

    def determine_consumption(self) -> float:
        return self.inflow * self.water_usage

    def determine_info(self) -> str:
        # TODO: Determine info for Power Plant.
        return ""
