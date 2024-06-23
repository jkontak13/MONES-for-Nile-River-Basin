import time
from pprint import pprint

import h5py
import numpy as np
import pandas
import torch
from matplotlib import pyplot as plt
from torch import nn

from comparison.converter import Converter
from core.learners.metrics import non_dominated
from core.learners.mones import MONES, run_episode
from datetime import datetime, date
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

# class Actor(nn.Module):
#     def __init__(self, nS, nA, hidden=50):
#         super(Actor, self).__init__()
#
#         self.nA = nA
#         self.fc1 = nn.Linear(nS, hidden)
#         self.fc2 = nn.Linear(hidden, nA)
#
#         nn.init.xavier_uniform_(self.fc1.weight, gain=1)
#         nn.init.xavier_uniform_(self.fc2.weight, gain=1)
#
#     def forward(self, state):
#
#         # actor
#         a = self.fc1(state.T)
#         a = torch.tanh(a)
#         a = self.fc2(a)
#         a = torch.relu(a)
#
#         return 100 * a

# class Actor(nn.Module):
#     def __init__(self, nS, nA, hidden=50):
#         super(Actor, self).__init__()
#
#         self.nA = nA
#         self.fc1 = nn.Linear(nS, hidden)
#         self.fc2 = nn.Linear(hidden, nA)
#
#         # nn.init.xavier_uniform_(self.fc1.weight, gain=1)
#         # nn.init.xavier_uniform_(self.fc2.weight, gain=1)
#         nn.init.uniform_(self.fc1.weight)
#         nn.init.uniform_(self.fc2.weight)
#
#     def forward(self, state):
#         # actor
#         a = self.fc1(state.T)
#         # a = torch.relu(a)
#         # a = torch.tanh(a)
#         a = self.fc2(a)
#         a = torch.relu(a)
#         return a * 1e-9


def train_agent(logdir, iterations=5, n_population=10, n_runs=1, parallel=False):
    epsilon = 1e-8

    ref_point_1_year = [0, -21107.50, -4710.52, 0]
    ref_point_2_years = [0, -42215.00, -9421.04, 0]
    ref_point_20_years = [0, -94210.39, -422150.04, 0]

    ref_point_20_years_weighted_deficit = [0, -2862500, -12847222.22, 0]

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
        ref_point=np.array(ref_point_20_years_weighted_deficit) + epsilon,
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
    tmp_agent = MONES(
        create_nile_river_env,
        Actor(5, 4, hidden=50),
        n_population=128,
        n_runs=1,
        logdir=None
    )

    state_dict = torch.load(logdir)
    tmp_agent.dist = state_dict['dist']
    tmp_agent.policy = state_dict['policy']

    # Sample agent population
    pop, _ = tmp_agent.sample_population()
    # Evaluate population
    r = tmp_agent.evaluate_population(create_nile_river_env(), pop, debug=True)
    print(r)
    return
    #
    # timesteps = 240
    # env = create_nile_river_env()
    # obs, _ = env.reset(seed=2137)
    # # print(obs)
    # reward = 0.0
    # gerd_water_level = []
    # for _ in range(timesteps):
    #     action = agent.forward(torch.from_numpy(obs).float()[:, None])
    #     action = action.detach().numpy().flatten()
    #     # action = [5, 5, 5, 5]
    #     print("Action:")
    #     pprint(action)
    #     (
    #         obs,
    #         final_reward,
    #         final_terminated,
    #         final_truncated,
    #         final_info,
    #     ) = env.step(action)
    #     print("Reward:")
    #     pprint(final_reward)
    #     # print("Info:")
    #     # pprint(final_info)
    #     gerd_water_level.append(final_info["GERD"]["current_level"])
    #     reward += final_reward
    # print("Final reward:", reward)
    # plt.plot(range(timesteps), gerd_water_level)
    # plt.show()


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

        print("Hypervolume:", group['hypervolume'][()])
        # print("Indicator metric:", group['metric'][()])
        nd_points = non_dominated(group["returns"]["ndarray"][-1])
        print("ND returns:", nd_points)
        print("ND returns converted:", Converter.convert_array(nd_points))
        # print(group['returns']['step'][()])
        print("Training took", group["time"][0][1], "seconds")

        plt.title("Hypervolume for the MONES training")
        plt.xlabel("Iteration number")
        plt.ylabel("Hypervolume")
        plt.plot(group["hypervolume"][()][:, 0], group["hypervolume"][()][:, 1], marker=".")
        plt.show()

def plot_sth():
    data = [587.862980304609, 585.8199598027771, 583.551263375436, 581.457895835866, 579.3892501038565, 578.1024412167004,
     581.561896090099, 590.2222751979162, 594.8809625157851, 596.3159152147392, 595.4468884436775, 593.857637151975,
     591.976563646656, 590.1918180038112, 587.7687322811062, 585.4905779329687, 583.462137864766, 582.4695410940233,
     585.3314061839532, 593.1161483152907, 597.660496326469, 598.9757085180038, 597.9936639897921, 596.2928822152835,
     594.3037155536351, 592.4244911777222, 590.3458975654082, 588.0368422745147, 585.8970589353993, 584.8060259376314,
     587.5738655240752, 594.8376397123942, 599.3139701827768, 600.3498771190222, 599.4238729781451, 597.660458318996,
     595.6105890940623, 593.6783076933758, 591.5427212013726, 589.4671572630674, 587.2648289844254, 586.1185013306263,
     588.8335239000983, 595.8046547639173, 600.1643870798522, 600.794840819214, 599.904926256852, 598.1204449181082,
     596.0501582805222, 594.1000310573485, 591.9452749112172, 589.9483065580614, 587.7249385407607, 586.5600100997368,
     589.2572653447502, 596.1299528311253, 600.3708820475367, 600.928378339519, 600.0235735238955, 598.2441746763839,
     596.1683960493855, 594.21346855745, 592.0535560509464, 590.0621867987827, 587.8487061628773, 586.6787741848206,
     589.3712500909365, 596.2174567097678, 600.425471508322, 600.9636806101254, 600.0467118916174, 598.2761557016116,
     596.198957533717, 594.2427892932461, 592.0815439974972, 590.0889497159521, 587.8806972288837, 586.709471952299,
     589.4007125099166, 596.2400744377032, 600.4395163980994, 600.9727632515809, 600.0526649788911, 598.2843469900888,
     596.2067852356956, 594.2502992027489, 592.0887125401706, 590.0958044922911, 587.8888911050991, 586.7173345757814,
     589.4082587234152, 596.2458675195513, 600.44310938848, 600.9750867901702, 600.0541879089853, 598.2864400763793,
     596.208785416222, 594.2522181790698, 592.090544288428, 590.097556065126, 587.8909848536791, 586.7193436806688,
     589.4101869773186, 596.2473478027031, 600.4440271847612, 600.9756803166899, 600.0545769274431, 598.2869745746556,
     596.2092961897378, 594.2527082159785, 592.0910120504549, 590.0980033532599, 587.8915195211229, 586.7198567331836,
     589.4106793834248, 596.2477258133277, 600.4442615405189, 600.9758318714088, 600.0546762618131, 598.287111042704,
     596.2094266003826, 594.2528333321548, 592.0911314794106, 590.0981175548231, 587.8916560323823, 586.7199877257208,
     589.4108051045221, 596.2478223269777, 600.4443213762169, 600.9758705663505, 600.0547016238586, 598.2871458857178,
     596.2094598968208, 594.2528652768095, 592.0911619720044, 590.0981467127596, 587.891690886429, 586.7200211707276,
     589.4108372036252, 596.2478469688372, 600.4443366534639, 600.9758804459408, 600.0547080992944, 598.2871547818343,
     596.2094683980655, 594.252873432917, 592.0911697573717, 590.0981541573623, 587.8916997853623, 586.7200297099048,
     589.4108453991664, 596.2478532603951, 600.4443405540494, 600.9758829683971, 600.0547097526024, 598.2871570531901,
     596.2094705686026, 594.2528755153337, 592.0911717451312, 590.0981560581175, 587.8917020574372, 586.7200318901268,
     589.4108474916513, 596.2478548667553, 600.4443415499468, 600.9758836124305, 600.054710174725, 598.2871576331127,
     596.2094711227842, 594.2528760470166, 592.091172252646, 590.0981565434186, 587.8917026375436, 586.7200324467814,
     589.4108480259047, 596.2478552768913, 600.4443418042194, 600.9758837768652, 600.0547102825012, 598.2871577811783,
     596.2094712642777, 594.2528761827657, 592.0911723822247, 590.0981566673256, 587.8917027856562, 586.7200325889064,
     589.4108481623103, 596.2478553816073, 600.4443418691403, 600.9758838188486, 600.0547103100188, 598.2871578189826,
     596.2094713004038, 594.252876217425, 592.0911724153085, 590.0981566989614, 587.8917028234722, 586.7200326251934,
     589.4108481971372, 596.2478554083432, 600.4443418857157, 600.9758838295677, 600.0547103170444, 598.2871578286347,
     596.2094713096275, 594.2528762262742, 592.0911724237554, 590.0981567070386, 587.8917028331273, 586.7200326344582,
     589.4108482060291, 596.2478554151693, 600.4443418899479, 600.9758838323046, 600.0547103188381, 598.2871578310991,
     596.2094713119826, 594.2528762285336, 592.0911724259122, 590.0981567091009, 587.8917028355925, 586.7200326368238,
     589.4108482082994, 596.2478554169122, 600.4443418910283, 600.9758838330033, 600.0547103192962, 598.2871578317281,
     596.2094713125837, 594.2528762291104, 592.0911724264627, 590.0981567096273, 587.8917028362217, 586.7200326374276,
     589.410848208879, 596.2478554173572, 600.4443418913041, 600.9758838331818, 600.0547103194132, 598.2871578318889]
    # plt.plot(pandas.date_range(date(2022,1, 1), date(2042,1, 1), freq="m"), data)
    plt.plot(range(1, 241), data)
    plt.ylim(top=640)
    plt.ylabel("Level (masl)")
    plt.xlabel("Months")
    plt.show()


if __name__ == "__main__":
    logdir = "runs/"
    logdir += datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + str(uuid.uuid4())[:4] + "/"

    train_agent(logdir, iterations=270, n_population=128, n_runs=1, parallel=True)

    # Trained agent path
    # temp = time.time()
    # logdir = "runs/2024-06-02_11-49-46_dfa5/checkpoint.pt"
    # logdir = "runs/2024-06-02_11-49-46_dfa5/checkpoint.pt"
    # run_agent(logdir)
    # print(time.time() - temp)
    # Read log file
    # logdir = "runs/2024-06-12_20-23-21_f2bd/log.h5"
    # show_logs(logdir)
    plot_sth()
