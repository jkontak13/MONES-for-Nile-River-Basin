from gymnasium import Space
from gymnasium.spaces import Box
from core.envs.water_management_system import WaterManagementSystem
from core.models.dam import Dam
from core.models.flow import Flow, Outflow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_system import IrrigationSystem
from core.models.catchment import Catchment
import pprint


def example_kris():
    # The numbers below are random and do not reflect actual constants of the Nile River simulation
    power_plant = PowerPlant("power-plant", 0.8, 1000, 1000, 1000, 0.2)
    dam = Dam("GERD", Box(0, 1000), Box(0, 1000), 1000)
    irrigation_system = IrrigationSystem("irrigation-system", 500)
    catchment = Catchment("catchment", 1000)

    power_plan_inflow = Inflow("inflow", power_plant, 1000, 1000)
    power_plant_dam_flow = Flow("power-plant-dam-flow", power_plant, dam, 1000)
    dam_irrigation_system_flow = Flow("dam_irrigation_system_flow", [dam, catchment], irrigation_system, 1000)
    irrigation_system_outflow = Outflow("outflow", dam, 1000)

    water_management_system = WaterManagementSystem(
        [
            power_plan_inflow,
            power_plant,
            power_plant_dam_flow,
            dam,
            dam_irrigation_system_flow,
            irrigation_system,
            irrigation_system_outflow,
        ]
    )


def example1():
    power_plant = PowerPlant("power-plant", 1000, 1000, 500, 100000, 0, Objective.identity, "ethiopia_power")
    dam = Dam("GERD", Space(), Box(0, 1000000), Objective.minimum_water_level, "min_HAD", stored_water=1000000000)
    irrigation_system = IrrigationSystem("irrigation-system", [100, 50, 1000], Objective.water_deficit_minimised, "egypt_deficit")

    power_plant_inflow = Inflow("inflow", power_plant, 1000, 1000)
    power_plant_dam_flow = Flow("power-plant-dam-flow", [power_plant], dam, 1000)
    dam_irrigation_system_flow = Flow("dam_irrigation_system_flow", [dam], irrigation_system, 1000)
    irrigation_system_outflow = Outflow("outflow", [irrigation_system], 1000)

    water_management_system = WaterManagementSystem(
        water_systems=[dam, dam_irrigation_system_flow, irrigation_system, irrigation_system_outflow],
        rewards={"ethiopia_power": 0, "egypt_deficit": 0, "min_HAD": 0},
        seed=2137,
    )
    for t in range(2):
        action = water_management_system.action_space.sample()
        print("Action:", action)
        final_observation, final_reward, final_terminated, final_truncated, final_info = water_management_system.step(action)
        print("Reward:", final_reward)
        pprint.pprint(final_info)
        print("Is finished:", final_truncated, final_terminated)


if __name__ == "__main__":
    example1()
