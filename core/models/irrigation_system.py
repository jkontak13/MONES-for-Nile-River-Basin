from core.models.facility import Facility
import numpy as np
import os

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(dir_path, "../data/")

"""
Class to represent Irrigation System

Attributes:
----------
id : str
    identifier
name : str
    Name of the Irrigation System
demand : float
    The demand of the irrigation system
deficit : float
   The total amount of water deficit we have 


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

    def __init__(self, id: str, name: str) -> None:
        super().__init__(id)
        self.name = name
        fh = os.path.join(data_directory, f"irr_demand_{name}.txt")
        self.demand = np.loadtxt(fh)
        self.deficit = 0
        self.months: int = 0

    """
        Calculates the reward (irrigation deficit) given the values of its attributes 

        Returns:
        ----------
        float
            Water deficit of the irrigation system
        """

    def determine_reward(self) -> float:
        consumption = self.determine_consumption()
        deficit = consumption - self.demand[self.months]
        self.deficit += deficit
        self.months += 1
        return deficit

    """
        Determines how much water is consumed by the irrigation system

            Returns:
            ----------
            float
                Water consumption
            """

    def determine_consumption(self) -> float:
        return min(self.demand[self.months], self.inflow-self.outflow)

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
            "name": self.name,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "demand": self.demand,
            "months": self.months,
            "deficit": self.deficit,
        }
