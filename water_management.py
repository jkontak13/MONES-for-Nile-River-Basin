import gymnasium
import h5py
import mo_gymnasium
import numpy as np
from pathlib import Path

import torch
from gymnasium import Space, ObservationWrapper
from gymnasium.spaces import Box
from torch import nn

from core.envs.water_management_system import WaterManagementSystem
from core.models.dam import Dam
from core.models.flow import Flow, Outflow, Inflow
from core.models.objective import Objective
from core.models.power_plant import PowerPlant
from core.models.irrigation_system import IrrigationSystem

import pygmo
from core.learners.mones import MONES
from datetime import datetime
import uuid
import torch.nn.functional as F


def make_env():
    env = mo_gymnasium.make("WaterManagementSystem-v0")


def make_nile_env():
    # Create power plant, dam and irrigation system. Initialise with semi-random parameters.
    # Set objective functions to identity for power plant, minimum_water_level for dam and water_deficit_minimised
    # for irrigation system.
    power_plant = PowerPlant("power-plant", Objective.identity, "ethiopia_power", 1000, 1000, 500, 100000, 1)
    dam = Dam("GERD", Box(low=0.0, high=10000000000), Box(0, 1000), Objective.minimum_water_level, "min_HAD", stored_water=5100000000)
    irrigation_system = IrrigationSystem(
        "irrigation-system", [100, 50, 1000], Objective.water_deficit_minimised, "egypt_deficit"
    )

    data_directory = Path(__file__).parent / "core" / "data"

    # Create 'edges' between Facilities.
    power_plant_inflow = Inflow(
        "inflow", power_plant, 1000000, np.loadtxt(data_directory / "catchments" / "white-nile.txt")
    )
    power_plant_dam_flow = Flow("power-plant-dam-flow", [power_plant], dam, 20000000)
    dam_irrigation_system_flow = Flow("dam_irrigation_system_flow", [dam], irrigation_system, 1000)
    irrigation_system_outflow = Outflow("outflow", [irrigation_system], 1000)

    # Create water management system. Add Facilities in the topological order (in the list).
    # Egypt deficit reward goes negative when there is a deficit. Otherwise is 0.
    water_management_system = WaterManagementSystem(
        water_systems=[
            power_plant_inflow,
            power_plant,
            power_plant_dam_flow,
            dam,
            dam_irrigation_system_flow,
            irrigation_system,
            irrigation_system_outflow,
        ],
        rewards={"ethiopia_power": 0, "egypt_deficit": 0, "min_HAD": 0},
        step_limit=12 * 20,
        seed=2137,
    )
    return water_management_system


class Actor(nn.Module):
    def __init__(self, nS, nA, hidden=50):
        super(Actor, self).__init__()

        self.nA = nA
        self.fc1 = nn.Linear(nS, hidden)
        self.fc2 = nn.Linear(hidden, nA)

        nn.init.xavier_uniform_(self.fc1.weight, gain=1)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1)

    def forward(self, state):

        # actor
        a = self.fc1(state)
        a = torch.tanh(a)
        a = self.fc2(a)
        return a


if __name__ == '__main__':
    logdir = 'runs/'
    logdir += datetime.now().strftime('%Y-%m-%d_%H-%M-%S_') + str(uuid.uuid4())[:4] + '/'

    agent = MONES(
        make_nile_env,
        Actor(1, 1, hidden=50),
        n_population=50,
        n_runs=10,
        logdir=logdir
    )

    agent.train(30)
    print("Dist:", agent.dist)
    print("Logdir:", logdir)
    torch.save({' dist': agent.dist, ' policy': agent.policy}, logdir + 'checkpoint.pt')

    f = h5py.File(Path(logdir) / 'log.h5', 'w')

    for key in f.keys():
        print(key)  # Names of the root level object names in HDF5 file - can be groups or datasets.
        print(type(f[key]))  # get the object type: usually group or dataset



