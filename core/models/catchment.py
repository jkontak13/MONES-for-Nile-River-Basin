from core.models.facility import Facility
from typing import List


class Catchment(Facility):

    def __init__(self, name: str, all_water_accumulated: List[float], timestep: int = 0) -> None:
        super().__init__(name, timestep=timestep)
        self.all_water_accumulated: List[float] = all_water_accumulated

    def determine_reward(self) -> float:
        return 0

    def determine_consumption(self) -> float:
        return -self.all_water_accumulated[self.timestep % len(self.all_water_accumulated)]

    def is_truncated(self) -> bool:
        return self.timestep >= len(self.all_water_accumulated)

    def determine_info(self) -> dict:
        return {"water_consumption": self.determine_consumption(), "timestep": self.timestep}
