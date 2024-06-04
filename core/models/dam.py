from pathlib import Path
from core.models.facility import ControlledFacility
from gymnasium.spaces import Box, Space
import numpy as np
from dateutil.relativedelta import relativedelta
from numpy.core.multiarray import interp as compiled_interp

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
    determine_outflow(action: float)
        Returns average monthly water release.
    """

    def __init__(
        self,
        name: str,
        observation_space: Space,
        action_space: Box,
        objective_function,
        integration_timestep_size: relativedelta,
        objective_name: str = "",
        max_capacity: float = float("Inf"),
        stored_water: float = 0,
    ) -> None:
        super().__init__(name, observation_space, action_space, max_capacity)
        self.stored_water: float = stored_water

        self.evap_rates = np.loadtxt(dam_data_directory / f"evap_{name}.txt")
        self.storage_to_minmax_rel = np.loadtxt(dam_data_directory / f"store_min_max_release_{name}.txt")
        self.storage_to_level_rel = np.loadtxt(dam_data_directory / f"store_level_rel_{name}.txt")
        self.storage_to_surface_rel = np.loadtxt(dam_data_directory / f"store_sur_rel_{name}.txt")

        self.storage_vector = []
        self.level_vector = []
        self.release_vector = []

        # Initialise storage vector
        self.storage_vector.append(stored_water)

        self.objective_function = objective_function
        self.objective_name = objective_name

        self.integration_timestep_size: relativedelta = integration_timestep_size

        # self.water_level = self.storage_to_level(self.stored_water)

    def determine_reward(self) -> float:
        # Pass water level to reward function
        return self.objective_function(self.storage_to_level(self.stored_water))

    def determine_outflow(self, action: float) -> float:
        current_storage = self.storage_vector[-1]
        in_month_releases = np.empty(0, dtype=np.float64)
        monthly_evap_total = 0

        final_date = self.current_date + self.timestep_size
        timestep_seconds = (final_date - self.current_date).total_seconds()
        evaporatio_rate_per_second = self.evap_rates[self.determine_month()] / (100 * timestep_seconds)

        while self.current_date < final_date:
            next_date = min(final_date, self.current_date + self.integration_timestep_size)
            integration_time_seconds = (next_date - self.current_date).total_seconds()
            self.current_date = next_date

            surface = self.storage_to_surface(current_storage)

            evaporation = surface * (evaporatio_rate_per_second * integration_time_seconds)
            monthly_evap_total += evaporation

            min_possible_release, max_possible_release = self.storage_to_minmax(current_storage)

            release_per_second = min(max_possible_release, max(min_possible_release, action))

            in_month_releases = np.append(in_month_releases, release_per_second)

            total_addition = self.get_inflow(self.timestep) * integration_time_seconds

            current_storage += total_addition - evaporation - release_per_second * integration_time_seconds

        # Update the amount of water in the Dam
        self.storage_vector.append(current_storage)
        self.stored_water = current_storage

        # Calculate the ouflow of water
        avg_monthly_release = np.mean(in_month_releases, dtype=np.float64)
        self.release_vector.append(avg_monthly_release)

        # Record level based on storage for time t
        self.level_vector.append(self.storage_to_level(current_storage))

        return avg_monthly_release

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

    def determine_month(self) -> int:
        return self.timestep % 12

    def storage_to_level(self, s: float) -> float:
        return self.modified_interp(s, self.storage_to_level_rel[0], self.storage_to_level_rel[1])

    def storage_to_surface(self, s: float) -> float:
        return self.modified_interp(s, self.storage_to_surface_rel[0], self.storage_to_surface_rel[1])

    def level_to_minmax(self, h) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.interp(h, self.rating_curve[0], self.rating_curve[1]),
            np.interp(h, self.rating_curve[0], self.rating_curve[2]),
        )

    def storage_to_minmax(self, s) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[1]),
            np.interp(s, self.storage_to_minmax_rel[0], self.storage_to_minmax_rel[2]),
        )

    @staticmethod
    def modified_interp(x: float, xp: float, fp: float, left=None, right=None) -> float:
        fp = np.asarray(fp)

        return compiled_interp(x, xp, fp, left, right)

    def reset(self) -> None:
        super().reset()
        stored_water = self.storage_vector[0]
        self.storage_vector = [stored_water]
        self.stored_water = stored_water
        self.level_vector = []
        self.release_vector = []
