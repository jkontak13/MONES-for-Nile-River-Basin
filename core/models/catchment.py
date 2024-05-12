from core.models.facility import Facility
from typing import List


class Catchment(Facility):

    def __init__(self, name: str, water_accumulated: List[float], timestep: int = 0) -> None:
        super().__init__(name, timestep=timestep)
        self.water_accumulated: float = water_accumulated

    def determine_reward(self) -> float:
        return 0

    def determine_consumption(self) -> float:
        return -self.water_accumulated

    def determine_info(self) -> dict:
        return {"water_accumulated": self.water_accumulated, "timestep": self.timestep}
