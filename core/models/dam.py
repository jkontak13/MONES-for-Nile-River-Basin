from core.models.facility import ControlledFacility
from gymnasium.core import ObsType


class Dam(ControlledFacility):
    def determine_reward(self) -> float:
        if self.stored_water > self.max_capacity:
            return float("-inf")
        else:
            return 0.0

    def determine_outflow(self, action) -> float:
        return self.inflow * action

    def determine_info(self) -> str:
        # TODO: Determine info for Dam.
        return ""

    def determine_observation(self) -> ObsType:
        # TODO: Determine observation.
        return self.stored_water

    def is_terminated(self) -> bool:
        return self.determine_reward() == float("-inf")
