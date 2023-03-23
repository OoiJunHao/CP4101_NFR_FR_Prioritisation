import numpy as np
from collections import defaultdict
from typing import List
import random as rnd
import json
from tqdm import tqdm

agent_num = 2000
search_iteration = 500
landscape_repetitions = 250


class LandScape_NFR_Introduction:
    def __init__(self, N, K, NFR_Count, M=None, K_within=None, K_between=None):
        self.N = N
        self.K = K
        self.M = M
        self.K_within = K_within
        self.K_between = K_between
        self.NFR_Count = NFR_Count
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
            for pos, j in enumerate(range(self.N, self.N + self.NFR_Count)):
                if i == pos:
                    IM[i][j] = 1
        for pos, i in enumerate(range(self.N, self.N + self.NFR_Count)):
            for j in range(0, self.N):
                if pos == j:
                    IM[i][j] = 1
        for i in range(self.N, self.N + self.NFR_Count):
            for j in range(self.N, self.N + self.NFR_Count):
                if i == j:
                    IM[i][j] = 1

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


class Agent_NFR_Introduction:
    def __init__(self, N, landscape, NFR_Count, introduction_step):
        self.N = N + NFR_Count
        self.state = np.random.choice([0, 1], self.N).tolist()
        self.landscape = landscape
        self.fitness = self.landscape.query_fitness(self.state)
        self.NFR_Count = NFR_Count
        self.introduction_step = introduction_step

        self.focus = (
            1  # 1 means to focus on FR, -1 means to do both FR and NFR at the same time
        )
        self.NFR_Range = [i for i in range(self.N - self.NFR_Count, self.N)]
        self.FR_Range = [i for i in range(self.N - self.NFR_Count)]

        self.count = 0

    def search(self):
        temp_state = list(self.state)
        if self.focus == 1:
            choice = np.random.choice(self.FR_Range)
            temp_state[choice] ^= 1
            if self.landscape.query_fitness(self.state) < self.landscape.query_fitness(
                temp_state
            ):
                self.state = temp_state
                self.fitness = self.landscape.query_fitness(temp_state)
            self.count += 1
            if self.count >= self.introduction_step:
                self.focus = -self.focus

        else:
            choice = np.random.choice(self.FR_Range)
            choice2 = choice + self.N - self.NFR_Count
            # temp_state[choice] ^= 1
            # temp_state[choice2] ^= 1
            # if self.landscape.query_fitness(self.state) < self.landscape.query_fitness(
            #     temp_state
            # ):
            #     self.state = temp_state
            #     self.fitness = self.landscape.query_fitness(temp_state)

            # experiment
            previous_fitness = self.landscape.query_fitness(self.state)

            temp_state_1 = list(self.state)
            temp_state_1[choice] ^= 1
            temp_state_1[choice2] ^= 1
            fitness_1 = self.landscape.query_fitness(temp_state_1)

            temp_state_2 = list(self.state)
            temp_state_2[choice] ^= 1
            fitness_2 = self.landscape.query_fitness(temp_state_2)

            temp_state_3 = list(self.state)
            temp_state_3[choice2] ^= 1
            fitness_3 = self.landscape.query_fitness(temp_state_3)

            if (
                fitness_1 > previous_fitness
                and fitness_1 >= fitness_2
                and fitness_1 >= fitness_3
            ):
                self.state = temp_state_1
                self.fitness = fitness_1
            elif (
                fitness_2 > previous_fitness
                and fitness_2 >= fitness_1
                and fitness_2 >= fitness_3
            ):
                self.state = temp_state_2
                self.fitness = fitness_2
            elif (
                fitness_3 > previous_fitness
                and fitness_3 >= fitness_2
                and fitness_3 >= fitness_1
            ):
                self.state = temp_state_3
                self.fitness = fitness_3


if __name__ == "__main__":
    ## Note: Due to unique struture of this landscape, N and NFR_Count must be equal
    N = 8
    NFR_Count = 8
    K = 3
    problem_spaces = {}
    for i in range(0, 160, 10):
        problem_spaces[f"{i}%"] = {
            "N": N,
            "NFR_Count": NFR_Count,
            "K": K,
            "introduction_step": int(i),
        }
    results = {}
    with tqdm(
        total=landscape_repetitions * len(problem_spaces)
    ) as pbar:  # for progress tracking
        for problem_space_name, problem_space_configs in problem_spaces.items():
            np.random.seed(100)
            agents_performance = []
            for i in range(landscape_repetitions):  # landscape repetitions
                landscape = LandScape_NFR_Introduction(
                    N=problem_space_configs["N"],
                    K=problem_space_configs["K"],
                    NFR_Count=problem_space_configs["NFR_Count"],
                )
                landscape.initialize(norm=True)
                # print(landscape.IM) # to remove
                agents = []
                for _ in range(agent_num):
                    agent = Agent_NFR_Introduction(
                        N=problem_space_configs["N"],
                        landscape=landscape,
                        NFR_Count=problem_space_configs["NFR_Count"],
                        introduction_step=problem_space_configs["introduction_step"],
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
                f"nfr_intro_N{N}K{K}_{problem_space_configs['introduction_step']}.csv",
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
            results[problem_space_name] = performance

    # output json
    json = json.dumps(results)
    f = open(f"nfr_intro_N{N}K{K}_results.json", "w")
    f.write(json)
    f.close()
