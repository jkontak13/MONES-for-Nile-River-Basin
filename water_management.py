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
        n_population=10,
        n_runs=5,
        logdir=logdir,
        indicator="hypervolume",
        # More thought needs to be put into this reference point. It seems to work for now.
        ref_point=np.array([-50000, -100000000, -100000000, -50000]),
    )

    agent.train(10)
    print("Distribution:", agent.dist)
    print("Logdir:", logdir)
    torch.save({" dist": agent.dist, " policy": agent.policy}, logdir + "checkpoint.pt")
