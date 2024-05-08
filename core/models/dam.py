from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space


class Dam(ControlledFacility):
    def __init__(
        self, id: str, observation_space: Space, action_space: Box, max_capacity: float, stored_water: float = 0
    ) -> None:
        super().__init__(id, observation_space, action_space, max_capacity)
        self.stored_water: float = stored_water

    def determine_reward(self) -> float:
        if self.stored_water > self.max_capacity:
            return float("-inf")
        else:
            return 0.0

    def determine_outflow(self, action: float) -> float:
        # TODO: Determine correct action type.
        return action

    def determine_info(self) -> str:
        # TODO: Determine info for Dam.
        return {}

    def determine_observation(self) -> float:
        # TODO: Determine observation.
        return self.stored_water

    def is_terminated(self) -> bool:
        return self.stored_water > self.max_capacity or self.stored_water < 0

    def step(self, action: float) -> tuple[float, float, bool, bool, dict]:
        self.outflow = self.determine_outflow(action)
        # TODO: Change stored_water to multiple outflows.
        self.stored_water += self.inflow - self.outflow

        return self.determine_observation(), self.determine_reward(), self.is_terminated(), False, self.determine_info()
