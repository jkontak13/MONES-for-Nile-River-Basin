from core.models.facility import ControlledFacility

class IrrigationSystem(ControlledFacility):
    def determine_reward(self) -> float:
        if self.stored_water > self.max_capacity:
            return float('-inf')
        else:
            return 0.0

    def step(self, action: float) -> float:
        # TODO: Change action to class so we can also release the water from storage
        self.outflow = self.inflow * action
        self.stored_water += self.inflow - self.outflow

        return self.determine_reward()
