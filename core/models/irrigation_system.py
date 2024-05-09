from core.models.facility import Facility
import numpy as np
from pathlib import Path


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

    def __init__(self, id: str, demand: list[float]) -> None:
        super().__init__(id)
        self.demand = demand
        self.total_deficit = 0
        self.months: int = 0
        self.list_deficits: list[float] = []

    """
    Calculates the reward (irrigation deficit) given the values of its attributes 

    Returns:
    ----------
    float
        Water deficit of the irrigation system
    """

    def determine_reward(self) -> float:
        consumption = self.determine_consumption()
        deficit = self.demand[self.months] - consumption
        self.total_deficit += deficit
        self.months += 1
        self.list_deficits.append(deficit)
        return deficit

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
