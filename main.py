import numpy as np
from pathlib import Path
import pprint
from gymnasium import Space
from gymnasium.spaces import Box
from core.envs.water_management_system import WaterManagementSystem
from core.models.dam import Dam
from core.models.flow import Flow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_system import IrrigationSystem
from core.models.catchment import Catchment


def nile_river_simulation(nu_of_timesteps=3):
    # Create power plant, dam and irrigation system. Initialise with semi-random parameters.
    # Set objective functions to identity for power plant, minimum_water_level for dam and water_deficit_minimised
    # for irrigation system.

    # Ethiopia
    GERD_dam = Dam(
        "GERD",
        Space(),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=15000000000,
    )
    GERD_power_plant = PowerPlant(
        "GERD_power_plant",
        Objective.identity,
        "ethiopia_power",
        efficiency=0.93,
        max_turbine_flow=4320,
        head_start_level=507,
        max_capacity=6000,
        dam=GERD_dam,
    )

    data_directory = Path(__file__).parent / "core" / "data"

    # Sudan
    DSSennar_irr_system = IrrigationSystem(
        "DSSennar_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_DSSennar.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Gezira_irr_system = IrrigationSystem(
        "Gezira_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Gezira.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Hassanab_irr_system = IrrigationSystem(
        "Hassanab_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Hassanab.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Tamaniat_irr_system = IrrigationSystem(
        "Tamaniat_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Tamaniat.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    USSennar_irr_system = IrrigationSystem(
        "USSennar_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_USSennar.txt"),
        Objective.water_deficit_minimised,
        "sudan_deficit_minimised",
    )
    Roseires_dam = Dam(
        "Roseires",
        Space(),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=4570000000,
    )
    Sennar_dam = Dam(
        "Sennar",
        Space(),
        Box(0, 10000),
        Objective.no_objective,
        stored_water=430000000,
    )

    # Egypt
    Egypt_irr_system = IrrigationSystem(
        "Egypt_irr",
        np.loadtxt(data_directory / "irrigation" / "irr_demand_Egypt.txt"),
        Objective.water_deficit_minimised,
        "egypt_deficit_minimised",
    )
    HAD_dam = Dam(
        "HAD",
        Space(),
        Box(0, 4000),
        Objective.minimum_water_level,
        "HAD_minimum_water_level",
        stored_water=137000000000,
    )

    # Create 'edges' between Facilities.
    # TODO: determine max capacity for flows

    GERD_inflow = Inflow(
        "gerd_inflow",
        GERD_dam,
        float("inf"),
        np.loadtxt(data_directory / "catchments" / "blue-nile.txt"),
    )

    # TODO: add catchment 1 inflow to sources of Roseires (inflow with destination Roseires)
    Roseires_flow = Flow("roseires_flow", [GERD_dam], Roseires_dam, float("inf"))

    # TODO: add catchment 2 inflow to sources of USSennar (inflow with destination USSennar)
    upstream_Sennar_received_flow = Flow(
        "upstream_Sennar_received_flow",
        [Roseires_dam],
        USSennar_irr_system,
        float("inf"),
    )

    # TODO: add catchment 3 inflow to sources of Sennar (inflow with destination USSennar)
    Sennar_flow = Flow("sennar_flow", [USSennar_irr_system], Sennar_dam, float("inf"))

    Gezira_received_flow = Flow("gezira_received_flow", [Sennar_dam], Gezira_irr_system, float("inf"))

    Dinder_catchment = Catchment("dinder_catchment", np.loadtxt(data_directory / "catchments" / "dinder.txt"))

    Rahad_catchment = Catchment("rahad_catchment", np.loadtxt(data_directory / "catchments" / "rahad.txt"))

    downstream_Sennar_received_flow = Flow(
        "downstream_sennar_received_flow",
        [Gezira_irr_system, Dinder_catchment, Rahad_catchment],
        DSSennar_irr_system,
        float("inf"),
    )

    WhiteNile_catchment = Catchment(
        "whitenile_catchment",
        np.loadtxt(data_directory / "catchments" / "white-nile.txt"),
    )

    Taminiat_received_flow = Flow(
        "taminiat_received_flow",
        [DSSennar_irr_system, WhiteNile_catchment],
        Tamaniat_irr_system,
        float("inf"),
    )

    Atbara_catchment = Catchment("atbara_catchment", np.loadtxt(data_directory / "catchments" / "atbara.txt"))

    # TODO: change Hassanab received flow to depend on leftover flow from Taminiat in previous month (see A.2.8)
    Hassanab_received_flow = Flow(
        "hassanab_received_flow",
        [Tamaniat_irr_system, Atbara_catchment],
        Hassanab_irr_system,
        float("inf"),
    )

    HAD_flow = Flow("had_flow", [Hassanab_irr_system], HAD_dam, float("inf"))

    Egypt_flow = Flow("egypt_flow", [HAD_dam], Egypt_irr_system, float("inf"))

    # Create water management system. Add Facilities in the topological order (in the list).
    # Egypt deficit reward goes negative when there is a deficit. Otherwise is 0.
    water_management_system = WaterManagementSystem(
        water_systems=[
            GERD_inflow,
            GERD_dam,
            GERD_power_plant,
            Roseires_flow,
            Roseires_dam,
            upstream_Sennar_received_flow,
            USSennar_irr_system,
            Sennar_flow,
            Sennar_dam,
            Gezira_received_flow,
            Gezira_irr_system,
            Dinder_catchment,
            Rahad_catchment,
            downstream_Sennar_received_flow,
            DSSennar_irr_system,
            WhiteNile_catchment,
            Taminiat_received_flow,
            Tamaniat_irr_system,
            Atbara_catchment,
            Hassanab_received_flow,
            Hassanab_irr_system,
            HAD_flow,
            HAD_dam,
            Egypt_flow,
            Egypt_irr_system,
        ],
        rewards={
            "ethiopia_power": 0,
            "sudan_deficit_minimised": 0,
            "egypt_deficit_minimised": 0,
            "HAD_minimum_water_level": 0,
        },
        seed=2137,
    )

    # Simulate for 3 timestamps (3 months).
    for _ in range(nu_of_timesteps):
        action = water_management_system.action_space.sample()
        print("Action:", action)
        (
            final_observation,
            final_reward,
            final_terminated,
            final_truncated,
            final_info,
        ) = water_management_system.step(action)
        print("Reward:", final_reward)
        pprint.pprint(final_info)
        print("Is finished:", final_truncated, final_terminated)


if __name__ == "__main__":
    nile_river_simulation()
