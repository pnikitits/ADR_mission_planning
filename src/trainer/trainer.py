import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque, namedtuple
import random
from torch.optim import AdamW

import os
import wandb

from src.agent.pytorch_agent import Agent
from src.rlglue.rl_glue import RLGlue
from src.rlglue.agent import BaseAgent
from src.environment.ADR_Environment import ADR_Environment
from src.visualisation.plot_script import plot_result

from tqdm import tqdm

def run_experiment(environment , agent , environment_parameters , agent_parameters , experiment_parameters):
    """
    Run the experiment
    """

    rl_glue = RLGlue(environment, agent)
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"],
                                 experiment_parameters["num_episodes"]))
    env_info = environment_parameters
    agent_info = agent_parameters

    

    for run in range(1 , experiment_parameters["num_runs"]+1):
        
        #changed_seed = 12 # override seed

        agent_info["seed"] = run # run
        agent_info["network_config"]["seed"] = run # run
        env_info["seed"] = run # run
        rl_glue.rl_init(agent_info , env_info)

        seed = agent_info["seed"]

        print(f"seed: {seed}")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        gpu_use = experiment_parameters['gpu_use']
        if gpu_use and torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        ep_count = 0
        for episode in tqdm(range(1 , experiment_parameters["num_episodes"]+1)):
            ep_count += 1
            #environment.pass_count(environment, message=f"Ep : {ep_count}")
            rl_glue.rl_episode(experiment_parameters["timeout"])

            # Get data from episode
            episode_reward = rl_glue.rl_agent_message("get_sum_reward")
            ep_loss = rl_glue.rl_agent_message(("get_loss"))
            fuel_limit, time_limit, impossible_dt, impossible_binary_flag = rl_glue.environment.get_term_reason()
            avg_fuel_used = rl_glue.environment.get_fuel_use_average()
            avg_time_used = rl_glue.environment.get_time_use_average()
            agent_sum_reward[run - 1, episode - 1] = episode_reward

            # wand logging
            wandb.log({
                    "episode loss": ep_loss,
                    "episode reward": episode_reward ,
                    "fuel limit": fuel_limit,
                    "time limit": time_limit,
                    "impossible_dt": impossible_dt,
                    "impossible_binary_flag": impossible_binary_flag,
                    "average fuel used":avg_fuel_used,
                    "average time used":avg_time_used,
                    "time step": episode
                })

    # print the average reward per run
    print("Average reward per run = ", np.mean(agent_sum_reward, axis=1))
    print("Overall average reward = ", np.mean(agent_sum_reward))

    wandb.log({"avg_reward": np.mean(agent_sum_reward)})


