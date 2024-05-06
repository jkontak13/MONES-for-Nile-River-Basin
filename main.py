from gymnasium.spaces import Box, Dict
from core.envs.water_management_system import WaterManagementSystem
from core.models.dam import Dam
from core.models.flow import Flow, Outflow, Inflow
from core.models.power_plant import PowerPlant
from core.models.irrigation_system import IrrigationSystem


power_plant = PowerPlant("power-plant", 1000, 0.2, 100)
dam = Dam("dam", Box(0, 1000), 1000)
irrigation_system = IrrigationSystem("irrigation-system", 1000, 500)

power_plan_inflow = Inflow("inflow", power_plant, 1000)
power_plant_dam_flow = Flow("power-plant-dam-flow", power_plant, dam, 1000)
dam_irrigation_system_flow = Flow("dam_irrigation_system_flow", dam, irrigation_system, 1000)
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
