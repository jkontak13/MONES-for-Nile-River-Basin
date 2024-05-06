from core.models.facility import Facility


class IrrigationSystem(Facility):

    def __init__(self, id: str, float, water_usage: float) -> None:
        super().__init__(id)
        self.water_usage: float = water_usage

    def determine_reward(self) -> float:
        return (self.outflow - self.inflow) / self.water_usage

    def determine_consumption(self) -> float:
        return min(self.water_usage, self.inflow)

    def determine_info(self) -> str:
        # TODO: Determine info for Irrigation System.
        return ""
