from typing import Tuple
from gymnasium.core import ObsType
from core.models.facility import Facility
import os

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(dir_path, "../data/")

"""
Class to represent Irrigation System

Attributes:
----------
id : str
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


class IrrigationSystem(Facility):

    def __init__(
        self,
        id: str,
        demand: list[float],
        objective_function,
        objective_name: str,
    ) -> None:
        super().__init__(id)
        self.demand = demand
        self.total_deficit = 0
        self.months: int = 0
        self.list_deficits: list[float] = []
        self.objective_function = objective_function
        self.objective_name = objective_name

    """
    Calculates the reward (irrigation deficit) given the values of its attributes 

    Returns:
    ----------
    float
        Water deficit of the irrigation system
    """

    def determine_deficit(self) -> float:
        consumption = self.determine_consumption()
        deficit = self.demand[self.months] - consumption
        self.total_deficit += deficit
        self.list_deficits.append(deficit)
        return deficit

    def determine_reward(self) -> float:
        """
        Calculates the reward given the objective function for this district.
        Uses consumption and demand.

        Returns:
        ----------
        float
            Reward for the objective function.
        """
        return self.objective_function(self.determine_consumption(), self.demand[self.months])

    """
    Determines how much water is consumed by the irrigation system

    Returns:
    ----------
    float
        Water consumption
    """

    def determine_consumption(self) -> float:
        return min(self.demand[self.months], self.inflow)

    """
    Determines info of irrigation system

    Returns:
    ----------
    dict
        Info about irrigation system (id, name, inflow, outflow, demand, months, deficit)
    """

    def determine_info(self) -> dict:
        return {
            "id": self.id,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "demand": self.demand,
            "months": self.months,
            "total_deficit": self.total_deficit,
            "list_deficits": self.list_deficits,
        }

    def step(self) -> Tuple[ObsType, float, bool, bool, dict]:
        # Get info and reward with the OLD month value
        info = self.determine_info()
        reward = self.determine_reward()

        # Increment the timestep
        self.months += 1
        return None, reward, False, False, info
