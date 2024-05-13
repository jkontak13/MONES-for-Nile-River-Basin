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

    def __init__(
        self,
        name: str,
        demand: List[float],
        objective_function,
        objective_name: str,
        timestep: int = 0,
    ) -> None:
        super().__init__(name, objective_function, objective_name, timestep)
        self.demand = demand
        self.total_deficit = 0
        self.list_deficits: List[float] = []

    def determine_deficit(self) -> float:
        """
        Calculates the reward (irrigation deficit) given the values of its attributes

        Returns:
        ----------
        float
            Water deficit of the irrigation system
        """
        consumption = self.determine_consumption()
        deficit = self.demand[self.timestep] - consumption
        self.total_deficit += deficit
        self.list_deficits.append(deficit)
        return deficit

    def determine_reward(self) -> float:
        """
        Calculates the reward given the objective function for this district.
        Uses demand and consumption.

        Returns:
        ----------
        float
            Reward for the objective function.
        """
        return self.objective_function(
            self.demand[self.timestep], self.determine_consumption()
        )

    def determine_consumption(self) -> float:
        """
        Determines how much water is consumed by the irrigation system

        Returns:
        ----------
        float
            Water consumption
        """
        return min(self.demand[self.timestep], self.inflow)

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
            "demand": self.demand,
            "timestep": self.timestep,
            "total_deficit": self.total_deficit,
            "list_deficits": self.list_deficits,
        }
