from core.models.facility import Facility
from typing import List


class IrrigationSystem(Facility):
    """
    Class to represent Irrigation System

    Attributes:
    ----------
    name : str
        identifier
    demand : float
    The list of monthly demand of the irrigation system
    total_deficit : float
    The total amount of water deficit we have
    list_deficits : list[float]
    The monthly list of the deficit of the irrigation system


    Methods:
    ----------
    determine_reward():
        Calculates the reward (irrigation deficit) given the values of its attributes
    determine_consumption():
        Determines how much water is consumed by the irrigation system
    determine_info():
        Returns info about the irrigation sustem
    """

    def __init__(self, name: str, all_demand: List[float], objective_function, objective_name: str) -> None:
        super().__init__(name, objective_function, objective_name)
        self.all_demand: List[float] = all_demand
        self.total_deficit = 0
        self.all_deficit: List[float] = []

    def get_current_demand(self) -> float:
        return self.all_demand[self.timestep % len(self.all_demand)]

    def determine_deficit(self) -> float:
        """
        Calculates the reward (irrigation deficit) given the values of its attributes

        Returns:
        ----------
        float
            Water deficit of the irrigation system
        """
        consumption = self.determine_consumption()
        deficit = self.get_current_demand() - consumption
        self.total_deficit += deficit
        self.all_deficit.append(deficit)
        return deficit

    def determine_reward(self) -> float:
        """
        Calculates the reward given the objective function for this district.
        Uses demand and received water.

        Returns:
        ----------
        float
            Reward for the objective function.
        """
        return self.objective_function(self.get_current_demand(), self.inflow, self.timestep)

    def determine_consumption(self) -> float:
        """
        Determines how much water is consumed by the irrigation system

        Returns:
        ----------
        float
            Water consumption
        """
        return min(self.get_current_demand(), self.inflow)

    def is_truncated(self) -> bool:
        return self.timestep >= len(self.all_demand)

    def determine_info(self) -> dict:
        """
        Determines info of irrigation system

        Returns:
        ----------
        dict
            Info about irrigation system (name, name, inflow, outflow, demand, timestep, deficit)
        """
        return {
            "name": self.name,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "demand": self.get_current_demand(),
            "total_deficit": self.total_deficit,
            "list_deficits": self.all_deficit,
        }

    def reset(self):
        super().reset()
        self.total_deficit = 0
        self.all_deficit = []
