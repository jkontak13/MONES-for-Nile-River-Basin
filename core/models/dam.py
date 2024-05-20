from typing import Tuple
from pathlib import Path
from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from numpy.core.multiarray import interp as compiled_interp
from array import array
from bisect import bisect_right
from exceptions import ReleaseRangeError

dam_data_directory = Path(__file__).parents[1] / "data" / "dams"


class Dam(ControlledFacility):
    """
    A class used to represent reservoirs/dams of the problem

    Attributes
    ----------
    name: str
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
    evap_rates: np.array (1x12)
        cm
        Monthly evaporation rates of the reservoir
    release_limits: dict
        Dict with release limits.
        Each key is a water level with corresponding release range

    Methods
    -------
    determine_current_level()
        Returns current water level
    determine_info()
        Return dictionary with parameters of the dam.
    determine_release_limit()
        Return min and max release based on current water level.
    storage_to_level(h=float)
        Returns the level(height) based on volume.
    level_to_storage(s=float)
        Returns the volume based on level(height).
    level_to_surface(h=float)
        Returns the surface area based on level.
    integration(
        total_seconds: int,
        release_action: float,
        net_inflow_per_second: float,
        integ_step: int,
        )
        Returns average monthly water release.
    """

    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: Box,
        objective_function,
        objective_name: str = "",
        max_capacity: float = float("Inf"),
        stored_water: float = 0,
        release_limits: dict = None,
    ) -> None:
        super().__init__(name, observation_space, action_space, max_capacity)
        self.stored_water: float = stored_water

        self.evap_rates = np.loadtxt(dam_data_directory / f"evap_{name}.txt")
        self.storage_to_minmax_rel = np.loadtxt(dam_data_directory / f"store_min_max_release_{name}.txt")
        self.storage_to_level_rel = np.loadtxt(dam_data_directory / f"store_level_rel_{name}.txt")
        self.storage_to_surface_rel = np.loadtxt(dam_data_directory / f"store_sur_rel_{name}.txt")

        self.storage_vector = array("f", [])
        self.level_vector = array("f", [])
        self.inflow_vector = array("f", [])
        self.release_vector = array("f", [])

        # Initialise storage vector
        self.storage_vector.append(stored_water)

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.release_limits = release_limits

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

    def determine_reward(self) -> float:
        return self.objective_function(self.stored_water)

    def determine_release_range(self) -> Tuple[float, float]:
        # If no release limits are specified at init, we return a range of [0, infinity]
        if not self.release_limits:
            return 0.0, float("inf")
        if len(self.level_vector) > 0:
            current_level = self.level_vector[-1]
        else:
            current_level = self.storage_to_level(self.stored_water)

        release_range = None
        max_level = 0

        # Water release limits are given by water level and corresponding min and max
        # Therefore we want to find the maximum water level that is smaller than the current water level
        for level in self.release_limits.keys():
            if (level >= max_level) and (level <= current_level):
                max_level = level
                release_range = self.release_limits[level]
        if release_range:
            return release_range[0], release_range[1]
        # When we do not find a release range that matches, we throw an error
        # This might happen when the current water level becomes too low, meaning no level will be below the current level
        else:
            raise ReleaseRangeError(
                "No release range found, either the release_limits attribute does not contain water levels, or the current water level is not valid.",
                self.name,
                current_level,
                self.release_limits,
            )

    def determine_outflow(self, action: float) -> float:
        # Clip action to release limits
        # (First wanted to implement this in ControlledFacility class but it doesn't have a level attribute)
        # Also looked into the option of using ActionWrapper but that is applied on gym.Env
        if self.release_limits:
            min_release, max_release = self.determine_release_range()
            # Check if function did not return None
            if min_release is not None and max_release is not None:
                # Clip action value between min and max release
                action = min(max_release, max(min_release, action))

        # TODO: Make timestep flexible for now it is hardcoded (one month)

        # Timestep is one month
        # Get the number of days in the month
        nu_days = self.nu_of_days_per_month[self.determine_month()]
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
            integ_step,
        )
        return outflow

    def determine_info(self) -> dict:
        info = {
            "name": self.name,
            "stored_water": self.stored_water,
            "current_level": self.level_vector[-1] if self.level_vector else None,
            "current_release": self.release_vector[-1] if self.release_vector else None,
            "evaporation_rates": self.evap_rates.tolist(),
        }
        return info

    def determine_observation(self) -> float:
        return self.stored_water

    def is_terminated(self) -> bool:
        return self.stored_water > self.max_capacity or self.stored_water < 0

    def determine_month(self):
        return self.timestep % 12

    def storage_to_level(self, s: float) -> float:
        return self.modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])

    def storage_to_surface(self, s: float) -> float:
        return self.modified_interp(s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1])

    def storage_to_minmax(self, s: float) -> Tuple[float, float]:
        # For minimum release constraint, we regard the data points as a step function
        # such that once a given storage/elevation is surpassed, we have to release a
        # certain given amount. For maximum, we use interpolation as detailed discharge
        # capacity calculations are made for certain points

        minimum_index = max(bisect_right(self.storage_to_minmax_rel[0], s), 1)
        minimum_cons = self.storage_to_minmax_rel[1][minimum_index - 1]
        maximum_cons = self.modified_interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2])

        return minimum_cons, maximum_cons

    def integration(
        self,
        total_seconds: int,
        release_action: float,
        net_inflow_per_second: float,
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
        release_action: float
            How much m3/s of water should be released.
        net_inflow_per_second: float
            Total inflow to this Dam measured in m3/s.
        integ_step: int
            Size of the integration step.

        Returns
        -------
        avg_monthly_release: float
            Average monthly release given in m3.
        """
        self.inflow_vector = np.append(self.inflow_vector, net_inflow_per_second)
        current_storage = self.storage_vector[-1]
        in_month_releases = array("f", [])
        monthly_evap_total = 0
        integ_step_count = total_seconds / integ_step

        for _ in np.arange(0, total_seconds, integ_step):
            surface = self.storage_to_surface(current_storage)

            evaporation = surface * (self.evap_rates[self.determine_month()] / (100 * integ_step_count))
            monthly_evap_total += evaporation

            min_possible_release, max_possible_release = self.storage_to_minmax(current_storage)

            release_per_second = min(max_possible_release, max(min_possible_release, release_action))

            in_month_releases.append(release_per_second)

            total_addition = net_inflow_per_second * integ_step

            current_storage += total_addition - evaporation - release_per_second * integ_step

        # Update the amount of water in the Dam
        self.storage_vector.append(current_storage)
        self.stored_water = current_storage

        # Calculate the ouflow of water
        avg_monthly_release = np.mean(in_month_releases)
        self.release_vector.append(avg_monthly_release)

        # Record level based on storage for time t
        self.level_vector.append(self.storage_to_level(current_storage))
        return avg_monthly_release

    @staticmethod
    def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
        fp = np.asarray(fp)

        return compiled_interp(x, xp, fp, left, right)
