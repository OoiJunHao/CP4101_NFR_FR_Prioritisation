import numpy as np
from collections import defaultdict
from typing import List
import random as rnd
import json
from tqdm import tqdm

agent_num = 1000
search_iteration = 250
landscape_repetitions = 100


# taking the basic landscape as the difference here is in the agent behaviour
class LandScape_FR_SizeComplexity:
    def __init__(
        self,
        N,
        K,
        NFR_Count,
        FR_NFR_K=0,
        NFR_FR_K=0,
        NFR_NFR_K=0,
        M=None,
        K_within=None,
        K_between=None,
    ):
        self.N = N
        self.K = K
        self.M = M
        self.K_within = K_within
        self.K_between = K_between
        self.NFR_Count = NFR_Count
        self.NFR_NFR_K = NFR_NFR_K
        self.FR_NFR_K = FR_NFR_K
        self.NFR_FR_K = NFR_FR_K
        self.IM, self.IM_dic = None, None
        self.FC = None
        self.cache = {}

    def create_influence_matrix(self):
        IM = np.eye(self.N + self.NFR_Count)

        # completely random
        if self.K_within is None:
            for i in range(self.N):
                probs = (
                    [1 / (self.N - 1)] * i + [0] + [1 / (self.N - 1)] * (self.N - 1 - i)
                )
                ids = np.random.choice(self.N, self.K, p=probs, replace=False)
                for index in ids:
                    IM[i][index] = 1
        # construct modularised landscape
        else:
            size_per_module = self.N // self.M
            for i in range(self.N):
                current_module = i // size_per_module
                within = [
                    j
                    for j in range(
                        current_module * size_per_module,
                        current_module * size_per_module + size_per_module,
                    )
                ]
                between = [
                    [j for j in range(0, current_module * size_per_module)],
                    [
                        j
                        for j in range(
                            current_module * size_per_module + size_per_module + 1,
                            self.N,
                        )
                    ],
                ]
                between = [
                    item for sublist in between for item in sublist
                ]  # flatten the list
                probs = (
                    [1 / (size_per_module - 1)] * (i % size_per_module)
                    + [0]
                    + [1 / (size_per_module - 1)]
                    * (size_per_module - 1 - (i % size_per_module))
                )
                ids_within = np.random.choice(
                    within, self.K_within, p=probs, replace=False
                )
                ids_between = np.random.choice(between, self.K_between, replace=False)
                for index in ids_within:
                    IM[i][index] = 1
                for index in ids_between:
                    IM[i][index] = 1
        # populate NFR-FR, FR-NFR, NFR-NFR values
        for i in range(0, self.N):
            probs = [1 / (self.NFR_Count)] * self.NFR_Count
            ids = np.random.choice(
                self.NFR_Count, self.FR_NFR_K, p=probs, replace=False
            )
            for index in ids:
                IM[i][index + self.N] = 1
        for i in range(self.N, self.N + self.NFR_Count):
            probs = [1 / (self.N)] * self.N
            ids = np.random.choice(self.N, self.NFR_FR_K, p=probs, replace=False)
            for index in ids:
                IM[i][index] = 1
        for i in range(self.N, self.N + self.NFR_Count):
            for j in range(self.N, self.N + self.NFR_Count):
                if i == j:
                    IM[i][j] = 1
            probs = (
                [1 / (self.NFR_Count - 1)] * (i - self.N)
                + [0]
                + [1 / (self.NFR_Count - 1)] * (self.NFR_Count - 1 - (i - self.N))
            )
            ids = np.random.choice(
                self.NFR_Count, self.NFR_NFR_K, p=probs, replace=False
            )
            for index in ids:
                IM[i][index + self.N] = 1
        IM_dic = defaultdict(list)
        for i in range(len(IM)):
            for j in range(len(IM[0])):
                if i == j or IM[i][j] == 0:
                    continue
                else:
                    IM_dic[i].append(j)
        self.IM, self.IM_dic = IM, IM_dic
        self.N = self.N + self.NFR_Count

    def create_fitness_config(
        self,
    ):
        FC = defaultdict(dict)
        for row in range(len(self.IM)):
            k = int(sum(self.IM[row]))
            for i in range(pow(2, k)):
                FC[row][i] = np.random.uniform(0, 1)
        self.FC = FC

    def calculate_fitness(self, state):
        res = 0.0
        for i in range(len(state)):
            dependency = self.IM_dic[i]
            bin_index = "".join([str(state[j]) for j in dependency])
            if state[i] == 0:
                bin_index = "0" + bin_index
            else:
                bin_index = "1" + bin_index
            index = int(bin_index, 2)
            res += self.FC[i][index]
        return res / len(state)

    def store_cache(
        self,
    ):
        for i in range(pow(2, self.N)):
            bit = bin(i)[2:]
            if len(bit) < self.N:
                bit = "0" * (self.N - len(bit)) + bit
            state = [int(cur) for cur in bit]
            self.cache[bit] = self.calculate_fitness(state)

    def initialize(self, first_time=True, norm=True):
        if first_time:
            self.create_influence_matrix()
        self.create_fitness_config()
        self.store_cache()

        # normalization
        if norm:
            normalizor = max(self.cache.values())
            min_normalizor = min(self.cache.values())

            for k in self.cache.keys():
                self.cache[k] = (self.cache[k] - min_normalizor) / (
                    normalizor - min_normalizor
                )
        self.cog_cache = {}

    def query_fitness(self, state):
        bit = "".join([str(state[i]) for i in range(len(state))])
        return self.cache[bit]


class Agent_Uncertainty:
    def __init__(self, N, NFR_Count, landscape, uncertainty, combined_search=False):
        self.N = N + NFR_Count
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.landscape = landscape
        self.fitness = self.landscape.query_fitness(self.state)
        self.NFR_Count = NFR_Count
        self.uncertainty = uncertainty
        self.NFR_Range = [i for i in range(self.N - self.NFR_Count, self.N)]

        self.search_space = [i for i in range(self.N - self.NFR_Count)]
        self.full_search_space = self.search_space.copy()
        self.combined_search = combined_search
        self.step = 1  # 1 means to look at FR, -1 means to look at NFR

    def search(self):
        # to do both NFR and FR in the same sprint
        if self.combined_search:
            temp_state = list(self.state)
            percieved_new_fitness = 0
            if self.step == 1:
                choice = np.random.choice(self.search_space)
                temp_state[choice] ^= 1
                error_term = np.random.normal(
                    0, self.uncertainty, 1
                )  # mean 0 and std based on supplied uncertainty
                percieved_new_fitness = (
                    self.landscape.query_fitness(temp_state) + error_term
                )
            else:
                choice = np.random.choice(self.NFR_Range)
                temp_state[choice] ^= 1
                percieved_new_fitness = self.landscape.query_fitness(temp_state)
            self.step = -self.step

            if self.landscape.query_fitness(self.state) < percieved_new_fitness:
                self.state = temp_state
                self.fitness = self.landscape.query_fitness(temp_state)

        # to do all FR first followed by all NFR
        else:
            temp_state = list(self.state)
            choice = (
                np.random.choice(self.search_space)
                if self.step == 1
                else np.random.choice(self.NFR_Range)
            )
            temp_state[choice] ^= 1

            percieved_new_fitness = 0
            if self.step == 1:
                error_term = np.random.normal(
                    0, self.uncertainty, 1
                )  # mean 0 and std based on supplied uncertainty
                percieved_new_fitness = (
                    self.landscape.query_fitness(temp_state) + error_term
                )
            else:
                percieved_new_fitness = self.landscape.query_fitness(temp_state)

            if self.landscape.query_fitness(self.state) < percieved_new_fitness:
                self.state = temp_state
                self.fitness = self.landscape.query_fitness(temp_state)
                self.search_space = self.full_search_space.copy()
            elif self.step == 1:
                self.search_space.remove(choice)
            if len(self.search_space) == 0:
                self.step = -1


if __name__ == "__main__":
    N = 12
    NFR_Count = 4
    problem_spaces = {
        "0.01": {
            "name": "0.01",
            "N": N,
            "NFR_Count": NFR_Count,
            "K": 5,
            "NFR_NFR_K": 1,
            "FR_NFR_K": 2,
            "NFR_FR_K": 6,
            "K_within": None,
            "K_between": None,
            "uncertainty": 0.01,
        },
        "0.05": {
            "name": "0.05",
            "N": N,
            "NFR_Count": NFR_Count,
            "K": 5,
            "NFR_NFR_K": 1,
            "FR_NFR_K": 2,
            "NFR_FR_K": 6,
            "K_within": None,
            "K_between": None,
            "uncertainty": 0.05,
        },
        "0.1": {
            "name": "0.1",
            "N": N,
            "NFR_Count": NFR_Count,
            "K": 5,
            "NFR_NFR_K": 1,
            "FR_NFR_K": 2,
            "NFR_FR_K": 6,
            "K_within": None,
            "K_between": None,
            "uncertainty": 0.1,
        },
    }
    results_together = {}
    with tqdm(
        total=landscape_repetitions * len(problem_spaces)
    ) as pbar:  # for progress tracking purposes
        for problem_space_name, problem_space_configs in problem_spaces.items():
            np.random.seed(100)
            agents_performance = []
            for i in range(landscape_repetitions):  # landscape repetitions
                landscape = LandScape_FR_SizeComplexity(
                    N=problem_space_configs["N"],
                    K=problem_space_configs["K"],
                    NFR_Count=problem_space_configs["NFR_Count"],
                    FR_NFR_K=problem_space_configs["FR_NFR_K"],
                    NFR_NFR_K=problem_space_configs["NFR_NFR_K"],
                    NFR_FR_K=problem_space_configs["NFR_FR_K"],
                    K_within=problem_space_configs["K_within"],
                    K_between=problem_space_configs["K_between"],
                )
                landscape.initialize(norm=True)
                # print(landscape.IM) # to remove
                agents = []
                for _ in range(agent_num):
                    agent = Agent_Uncertainty(
                        N=problem_space_configs["N"],
                        landscape=landscape,
                        NFR_Count=problem_space_configs["NFR_Count"],
                        uncertainty=problem_space_configs["uncertainty"],
                        combined_search=True,
                    )
                    agents.append(agent)
                for agent in agents:  # agent repetitions
                    agent_performance = []
                    for _ in range(search_iteration):  # i.e., the number of steps
                        agent.search()
                        agent_performance.append(agent.fitness)
                    agents_performance.append(agent_performance)
                pbar.update(1)
            np.savetxt(
                f"uncertainty__{problem_space_configs['name']}__together.csv",
                agents_performance,
                delimiter=",",
            )  # save to csv for analysis
            performance = []
            for period in range(search_iteration):
                temp = [
                    agent_performance[period]
                    for agent_performance in agents_performance
                ]
                performance.append(sum(temp) / len(temp))
            results_together[problem_space_name] = performance

    # output json
    json_together = json.dumps(results_together)
    f = open(f"uncertainty__N{N}__together__results.json", "w")
    f.write(json_together)
    f.close()

    results_separate = {}
    with tqdm(
        total=landscape_repetitions * len(problem_spaces)
    ) as pbar:  # for progress tracking purposes
        for problem_space_name, problem_space_configs in problem_spaces.items():
            np.random.seed(100)
            agents_performance = []
            for i in range(landscape_repetitions):  # landscape repetitions
                landscape = LandScape_FR_SizeComplexity(
                    N=problem_space_configs["N"],
                    K=problem_space_configs["K"],
                    NFR_Count=problem_space_configs["NFR_Count"],
                    FR_NFR_K=problem_space_configs["FR_NFR_K"],
                    NFR_NFR_K=problem_space_configs["NFR_NFR_K"],
                    NFR_FR_K=problem_space_configs["NFR_FR_K"],
                    K_within=problem_space_configs["K_within"],
                    K_between=problem_space_configs["K_between"],
                )
                landscape.initialize(norm=True)
                # print(landscape.IM) # to remove
                agents = []
                for _ in range(agent_num):
                    agent = Agent_Uncertainty(
                        N=problem_space_configs["N"],
                        landscape=landscape,
                        NFR_Count=problem_space_configs["NFR_Count"],
                        uncertainty=problem_space_configs["uncertainty"],
                        combined_search=False,
                    )
                    agents.append(agent)
                for agent in agents:  # agent repetitions
                    agent_performance = []
                    for _ in range(search_iteration):  # i.e., the number of steps
                        agent.search()
                        agent_performance.append(agent.fitness)
                    agents_performance.append(agent_performance)
                pbar.update(1)
            np.savetxt(
                f"uncertainty__{problem_space_configs['name']}__separate.csv",
                agents_performance,
                delimiter=",",
            )  # save to csv for analysis
            performance = []
            for period in range(search_iteration):
                temp = [
                    agent_performance[period]
                    for agent_performance in agents_performance
                ]
                performance.append(sum(temp) / len(temp))
            results_separate[problem_space_name] = performance

    # output json
    json_separate = json.dumps(results_separate)
    f = open(f"availability__N{N}__separate__results.json", "w")
    f.write(json_separate)
    f.close()
