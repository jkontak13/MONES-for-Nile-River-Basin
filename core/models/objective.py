class Objective:
    MINIMUM_WATER_LEVEL = 159

    num_of_days_in_month = [
            31,  # January
            28,  # February (non-leap year)
            31,  # March
            30,  # April
            31,  # May
            30,  # June
            31,  # July
            31,  # August
            30,  # September
            31,  # October
            30,  # November
            31,  # December
        ]

    @staticmethod
    def no_objective(*args):
        return 0.0

    @staticmethod
    def identity(value: float) -> float:
        return value

    @staticmethod
    def minimum_water_level(water_level: float) -> float:
        return 0.0 if water_level < Objective.MINIMUM_WATER_LEVEL else 1.0

    @staticmethod
    def water_deficit_minimised(demand: float, received: float) -> float:
        return -max(0.0, demand - received)

    SCALAR = 1000000000

    @staticmethod
    def scalar_identity(value: float) -> float:
        return value / Objective.SCALAR

    @staticmethod
    def water_deficit_minimised_weighted(demand: float, received: float, timestep: int) -> float:
        return -max(0.0, demand - received) * Objective.num_of_days_in_month[timestep % 12]
