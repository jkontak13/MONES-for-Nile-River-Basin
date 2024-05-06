from core.models.facility import Facility

"""
Class to represent Hydro-energy Powerplant

Attributes:
----------
id : str
    identifier
efficiency : float
    Efficiency coefficient (mu) used in hydropower formula
max_turbine_flow : float
    Maximum possible flow that can be passed through the turbines for the
    purpose of hydroenergy production
head_start_level : float
    Minimum elevation of water level that is used to calculate hydraulic
    head for hydropower production
max_capacity : float
    Total design capacity (mW) of the plant
water_level_coeff : float
    Coefficient that determines the water level based on the volume of outflow
    Used to calculate at what level the head of the power plant operates
operating_hours : float
    Amount of hours that the plant operates, used to calculate power generation
water_usage : float
    Amount of water  that is used by plant, decimal coefficient

Methods:
----------
determine_reward():
    Calculates the reward (power generation) given the values of its attributes 
determine_consumption():
    Determines how much water is consumed by the power plant
determine_info():
    Returns info about the hydro-energy powerplant
"""


class PowerPlant(Facility):
    def __init__(
        self,
        id: str,
        efficiency: float,
        max_turbine_flow: float,
        head_start_level: float,
        max_capacity: float,
        water_level_coeff: float,
        # TODO: find out if operating hours for power plants differ
        operating_hours: float = 24 * 30,
        # TODO: determine actual water usage for power plants, 0.0 for ease now
        water_usage: float = 0.0,
    ) -> None:
        super().__init__(id)
        self.efficiency = efficiency
        self.max_turbine_flow = max_turbine_flow
        self.head_start_level = head_start_level
        self.max_capacity = max_capacity
        self.water_level_coeff = water_level_coeff
        self.operating_hours = operating_hours
        self.water_usage = water_usage
        # Create value to track number of months and total production, can be used to track yearly or montly averages
        self.months = 0
        self.production_sum = 0

    """
    Calculates the reward for a power plant.
    Currently only calculates power production in MWh

    Parameters:
    ----------
    m3_to_kg_factor : int
        Factor to multiply by, to go from m3 to kg
    w_mw_conversion : float
        Factor to convert W to mW
    g : float
        Gravity constant

    Returns:
    ----------
    float
        Plant's power production in mWh
    """

    # Constants are configured as parameters with default values
    def determine_reward(
        self,
        m3_to_kg_factor: int = 1000,
        w_mw_conversion: float = 1e-6,
        g: float = 9.81,
    ) -> float:
        # Turbine flow is equal to outflow, as long as it does not exceed maximum turbine flow
        turbine_flow = min(self.outflow, self.max_turbine_flow)

        # Uses water level coeff to calculate at what level the outflow will flow
        water_level = self.outflow * self.water_level_coeff
        # Calculate at what level the head will generate power, using water_level of the outflow and head_start_level
        head = max(0, water_level - self.head_start_level)

        # Calculate power in mW, has to be lower than or equal to capacity
        power_in_mw = min(
            self.max_capacity,
            turbine_flow * head * m3_to_kg_factor * g * self.efficiency * w_mw_conversion,
        )

        # Hydro-energy power production in mWh
        production = power_in_mw * self.operating_hours

        # TODO: change function for hydro-energy power production to be an actual reward
        #       (adapt it to the specific objective such as minimising months below minimum power production)
        #       possibly use months and production_sum attributes to calculate monthly or yearly averages
        self.months += 1
        self.production_sum += production

        return production

    """
    Determines water consumption.

    Returns:
    ----------
    float
        How much water is consumed
    """

    def determine_consumption(self) -> float:
        return self.inflow * self.water_usage

    """
    Determines info of hydro-energy power plant

    Returns:
    ----------
    dict
        Info about power plant (id, inflow, outflow, water usage, months, total production)
    """

    def determine_info(self) -> dict:
        return {
            "id": self.id,
            "inflow": self.inflow,
            "outflow": self.outflow,
            "water_usage": self.water_usage,
            "months": self.months,
            "total production (MWh)": self.production_sum,
        }
