import numpy as np
import torch
from torch import nn

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
        a = self.fc1(state)
        a = torch.tanh(a)
        a = self.fc2(a)
        return a


if __name__ == "__main__":
    logdir = "runs/"
    logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

    number_of_objectives = 4
    number_of_actions = 4
    agent = MONES(
        create_nile_river_env,
        Actor(number_of_objectives, number_of_actions, hidden=50),
        n_population=1000,
        n_runs=10,
        logdir=logdir,
        indicator="hypervolume",
        ref_point=np.array([-50000, -50000, -100000, -50000]),
    )

    agent.train(100)
    print("Distribution:", agent.dist)
    print("Logdir:", logdir)
    torch.save({" dist": agent.dist, " policy": agent.policy}, logdir + "checkpoint.pt")
