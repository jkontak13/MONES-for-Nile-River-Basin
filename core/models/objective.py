class Objective:

    @staticmethod
    def no_objective(*args):
        return 0.0

    @staticmethod
    def identity(value: float) -> float:
        return value

    @staticmethod
    def minimum_water_level(minimum_water_level: float) -> float:
        return lambda water_level: 0.0 if water_level < minimum_water_level else 1.0

    @staticmethod
    def water_deficit_minimised(demand: float, received: float) -> float:
        return -max(0.0, demand - received)

    @staticmethod
    def supply_ratio_maximised(demand: float, received: float) -> float:
        return received / demand

    @staticmethod
    def scalar_identity(scalar: float) -> float:
        return lambda value: value * scalar
