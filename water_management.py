import time
from pprint import pprint

import h5py
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from core.learners.metrics import non_dominated
from core.learners.mones import MONES
from datetime import datetime
import uuid
from main import create_nile_river_env


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
        a = self.fc1(state.T)
        a = torch.tanh(a)
        a = self.fc2(a)
        return 100 * a


def train_agent(logdir, iterations=5, n_population=10, n_runs=1, parallel=False):
    epsilon = 1e-8

    ref_point_1_year = [0, -21107.50, -4710.52, 0]
    ref_point_2_years = [0, -42215.00, -9421.04, 0]
    ref_point_20_years = [0, -94210.39, -422150.04, 0]
    number_of_observations = 5
    number_of_actions = 4
    agent = MONES(
        create_nile_river_env,
        Actor(number_of_observations, number_of_actions, hidden=50),
        n_population=n_population,
        n_runs=n_runs,
        logdir=logdir,
        indicator="hypervolume",
        # TODO: Change depending on the time horizon
        ref_point=np.array(ref_point_20_years) + epsilon,
        parallel=parallel,
    )
    timer = time.time()
    agent.train(iterations)
    agent.logger.put("train/time", time.time() - timer, 0, "scalar")
    print(f"Training took: {time.time() - timer} seconds")

    print("Logdir:", logdir)
    torch.save({"dist": agent.dist, "policy": agent.policy}, logdir + "checkpoint.pt")


def run_agent(logdir):
    # Load agent
    checkpoint = torch.load(logdir)
    print(checkpoint)
    agent = checkpoint["policy"]

    timesteps = 240
    env = create_nile_river_env()
    obs, _ = env.reset(seed=2137)
    # print(obs)
    for _ in range(timesteps):
        action = agent.forward(torch.from_numpy(obs).float())
        action = action.detach().numpy().flatten()
        # action = [5, 5, 5, 5]
        # print("Action:")
        # pprint(action)
        (
            final_observation,
            final_reward,
            final_terminated,
            final_truncated,
            final_info,
        ) = env.step(action)
        # print("Reward:")
        # pprint(final_reward)
        # print("Observation:")
        # pprint(final_observation)


def show_logs(logdir):
    with h5py.File(logdir, "r") as f:
        # Print all root level object names (aka keys)
        # these can be group or dataset names
        print("Keys: %s" % f.keys())
        # get first object name/key; may or may NOT be a group
        a_group_key = list(f.keys())[0]

        # get the object type for a_group_key: usually group or dataset
        print(type(f[a_group_key]))

        # If a_group_key is a group name,
        # this gets the object names in the group and returns as a list
        data = list(f[a_group_key])
        print(data)
        print(list(f[list(f.keys())[1]]))

        params = f["params"]

        print("Iterations: \t", params["iterations"][0][1])
        print("N_populations: \t", params["n_population"][0][1])
        print("Parallel: \t", params["parallel"][0][1])

        group = f["train"]

        # print("Hypervolume:", group['hypervolume'][()])
        # print("Indicator metric:", group['metric'][()])
        print("ND returns:", non_dominated(group["returns"]["ndarray"][-1]))
        # print(group['returns']['step'][()])
        print("Training took", group["time"][0][1], "seconds")

        plt.plot(group["hypervolume"][()][:, 0], group["hypervolume"][()][:, 1], marker=".")
        plt.show()


if __name__ == "__main__":
    logdir = "runs/"
    logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

    train_agent(logdir, iterations=50, n_population=5, n_runs=1, parallel=True)

    # Trained agent path
    # temp = time.time()
    # logdir = "runs/2024-05-21_19-09-55_ac09/checkpoint.pt"
    # run_agent(logdir)
    # print(time.time() - temp)
    # Read log file
    # logdir = "runs/2024-05-27_17-10-34_3b0f/log.h5"
    # show_logs(logdir)
