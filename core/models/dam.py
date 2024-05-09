from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from numpy.core.multiarray import interp as compiled_interp
import os
from scipy.constants import g
from array import array
from bisect import bisect_right

dir_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
data_directory = os.path.join(dir_path, "../data/")


def modified_interp(x, xp, fp, left=None, right=None):
    fp = np.asarray(fp)

    interp_func = compiled_interp
    return interp_func(x, xp, fp, left, right)


class Dam(ControlledFacility):
    """
    A class used to represent reservoirs/dams of the problem

    Attributes
    ----------
    id: str
        Lowercase non-spaced name of the reservoir
    storage_vector: np.array (1xH)
        m3
        A vector that holds the volume of the water in the reservoir
        throughout the simulation horizon
    level_vector: np.array (1xH)
        m
        A vector that holds the elevation of the water in the reservoir
        throughout the simulation horizon
    release_vector: np.array (1xH)
        m3/s
        A vector that holds the actual average release per month
        from the reservoir throughout the simulation horizon
    hydropower_plant : HydropowerPlant
        A hydropower plant object belonging to the reservoir
    hydroenergy_produced: np.array (1xH)
        MWh
        Amount of hydroenergy produced in each month
    evap_rates: np.array (1x12)
        cm
        Monthly evaporation rates of the reservoir

    Methods
    -------
    determine_info()
        Return dictionary with parameters of the dam.
    storage_to_level(h=float)
        Returns the level(height) based on volume.
    level_to_storage(s=float)
        Returns the volume based on level(height).
    level_to_surface(h=float)
        Returns the surface area based on level.
    integration(total_seconds: int,
        policy_release_decision: float,
        net_secondly_inflow: float,
        current_month: int,
        integ_step: int,
        )
        Returns average monthly water release.
    """

    def __init__(
        self,
        id: str,
        observation_space: Space,
        action_space: Box,
        max_capacity: float,
        objective_function,
        objective_name: str = "",
        stored_water: float = 0,
    ) -> None:
        super().__init__(id, observation_space, action_space, max_capacity)
        self.stored_water: float = stored_water

        fh = os.path.join(data_directory, f"evap_{id}.txt")
        self.evap_rates = np.loadtxt(fh)

        fh = os.path.join(data_directory, f"store_min_max_release_{id}.txt")
        self.storage_to_minmax_rel = np.loadtxt(fh)

        fh = os.path.join(data_directory, f"store_level_rel_{id}.txt")
        self.storage_to_level_rel = np.loadtxt(fh)

        fh = os.path.join(data_directory, f"store_sur_rel_{id}.txt")
        self.storage_to_surface_rel = np.loadtxt(fh)

        self.storage_vector = array("f", [])
        self.level_vector = array("f", [])
        self.inflow_vector = array("f", [])
        self.release_vector = array("f", [])
        self.hydropower_plant = None
        self.hydroenergy_produced = array("f", [])
        # self.total_evap = np.empty(0)

        # Initialise storage vector
        self.storage_vector.append(stored_water)

        self.objective_function = objective_function
        self.objective_name = objective_name

        # TODO: Read it from file
        self.nu_of_days_per_month = [
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
        self.months = 0

    def determine_reward(self) -> float:
        return self.objective_function(self.stored_water)

    def determine_outflow(self, action: float) -> float:
        # TODO: Make timestep flexible for now it is hardcoded (one month)

        # Timestep is one month
        # Get the number of days in the month
        nu_days = self.nu_of_days_per_month[self.months]
        # Calculate hours
        total_hours = nu_days * 24
        # Calculate seconds
        total_seconds = total_hours * 3600
        # Calculate integration step
        integ_step = total_seconds / (nu_days * 48)
        # Calculate outflow using integration function
        outflow = self.integration(
            total_seconds,
            action,
            self.inflow,
            self.months,
            integ_step,
        )
        return outflow

    def determine_info(self) -> dict:
        info = {
            "id": self.id,
            "stored_water": self.stored_water,
            "current_level": self.level_vector[-1] if self.level_vector else None,
            "current_release": self.release_vector[-1] if self.release_vector else None,
            "current_hydroenergy_produced": self.hydroenergy_produced[-1] if self.hydroenergy_produced else None,
            "evaporation_rates": self.evap_rates.tolist(),
            "max_capacity": self.max_capacity,
        }
        return info

    def determine_observation(self) -> float:
        return self.stored_water

    def is_terminated(self) -> bool:
        return self.stored_water > self.max_capacity or self.stored_water < 0

    def step(self, action: float) -> tuple[float, float, bool, bool, dict]:
        # Determine outflow (determine_outflow->integration() updates the stored_water variable)
        self.outflow = self.determine_outflow(action)
        # Increase the month number by one
        self.months = (self.months + 1) % 12

        return self.determine_observation(), self.determine_reward(), self.is_terminated(), False, self.determine_info()

    def storage_to_level(self, s: float):
        return modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])

    def storage_to_surface(self, s: float):
        return modified_interp(s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1])

    def storage_to_minmax(self, s: float):
        # For minimum release constraint, we regard the data points as a step function
        # such that once a given storage/elevation is surpassed, we have to release a
        # certain given amount. For maximum, we use interpolation as detailed discharge
        # capacity calculations are made for certain points

        minimum_index = max(bisect_right(self.storage_to_minmax_rel[0], s), 1)
        minimum_cons = self.storage_to_minmax_rel[1][minimum_index - 1]
        maximum_cons = modified_interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2])

        return minimum_cons, maximum_cons

    def integration(
        self,
        total_seconds: int,
        policy_release_decision: float,
        net_secondly_inflow: float,
        current_month: int,
        integ_step: int,
    ) -> float:
        """
        Converts the flows of the reservoir into storage. Time step
        fidelity can be adjusted within a for loop. The core idea is to
        arrive at m3 storage from m3/s flows.

        Parameters
        ----------
        total_seconds: int
            Number of seconds in the timestep.
        policy_release_decision: float
            How much m3/s of water should be released.
        net_secondly_inflow: float
            Total inflow to this Dam.
        current_month: int
            Current month.
        integ_step: int
            Size of the integration step.

        Returns
        -------
        avg_monthly_release

        """

        self.inflow_vector = np.append(self.inflow_vector, net_secondly_inflow)
        current_storage = self.storage_vector[-1]
        in_month_releases = array("f", [])
        monthly_evap_total = 0
        integ_step_count = total_seconds / integ_step

        for _ in np.arange(0, total_seconds, integ_step):

            surface = self.storage_to_surface(current_storage)

            evaporation = surface * (self.evap_rates[current_month - 1] / (100 * integ_step_count))
            monthly_evap_total += evaporation

            min_possible_release, max_possible_release = self.storage_to_minmax(current_storage)

            secondly_release = min(max_possible_release, max(min_possible_release, policy_release_decision))

            in_month_releases.append(secondly_release)

            total_addition = net_secondly_inflow * integ_step

            current_storage += total_addition - evaporation - secondly_release * integ_step

        # Update the amount of water in the Dam
        self.storage_vector.append(current_storage)
        self.stored_water = current_storage

        # Calculate the ouflow of water
        avg_monthly_release = np.mean(in_month_releases)
        self.release_vector.append(avg_monthly_release)

        # self.total_evap = np.append(self.total_evap, monthly_evap_total)

        # Record level based on storage for time t
        self.level_vector.append(self.storage_to_level(current_storage))

        return avg_monthly_release
