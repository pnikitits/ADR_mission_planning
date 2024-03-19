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

def run_experiment(environment , agent , environment_parameters , agent_parameters , experiment_parameters):
    """
    Run the experiment
    """

    rl_glue = RLGlue(environment, agent)
    agent_sum_reward = np.zeros((experiment_parameters["num_runs"],
                                 experiment_parameters["num_episodes"]))
    env_info = {}
    agent_info = agent_parameters
    for run in range(1 , experiment_parameters["num_runs"]+1):
        agent_info["seed"] = run
        agent_info["network_config"]["seed"] = run
        env_info["seed"] = run
        rl_glue.rl_init(agent_info , env_info)

        seed = agent_info["seed"]
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
        for episode in range(1 , experiment_parameters["num_episodes"]+1):
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
                    "average time used":avg_time_used
                })

    wandb.log({"avg_reward": sum(agent_sum_reward[0])/experiment_parameters["num_episodes"]})
 
    # subfolder = 'models/'
    # model_name = input("model name : ")
    # if not os.path.exists("models"):
    #     os.makedirs("models")
    # if model_name != '':
    #     torch.save(rl_glue.agent.policy_network.state_dict(), subfolder + model_name +'.pth')
    #     print("Model saved as ", model_name)

    # save_name = "{}".format(rl_glue.agent.name)
    # if not os.path.exists("results"):
    #     os.makedirs("results")
    # np.save("results/sum_reward_{}".format(save_name) , agent_sum_reward)





if __name__ == "__main__":

    weights_file = None #'models/test_weights.pth'
    experiment_parameters = {"num_runs":1,
                            "num_episodes":50,
                            "timeout":20,
                            "gpu_use":False,
                            "track_wandb":True}
    environment_parameters = {}
    current_env = ADR_Environment
    agent_parameters = {"network_config":{"state_dim":25,
                                        "num_hidden_units":512,
                                        "num_actions":300,
                                        "weights_file":weights_file},
                        "optimizer_config":{"step_size":1e-4, # working value 1e-3
                                            "beta_m":0.9,
                                            "beta_v":0.999,
                                            "epsilon":1e-8},
                        "replay_buffer_size":1000000,
                        "minibatch_size":64,
                        "num_replay_updates_per_step":4,
                        "gamma":0.99,
                        "tau":0.001,
                        "seed":0
                        }

    # Set device
    gpu_use = experiment_parameters['gpu_use']
    if gpu_use and torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using the GPU")
    elif gpu_use and not torch.cuda.is_available():
        device = torch.device("cpu")
        print('GPU is not available. Using CPU')
    else:
        device = torch.device("cpu")
        print("Using the CPU")

    agent_parameters['device'] = device
    current_agent = Agent

    # Setup wandb
    global track_wandb
    track_wandb = experiment_parameters['track_wandb']
    if track_wandb:
        wandb.login()
        wandb.init(
            project="ADR Mission Planning",
            config = agent_parameters
            )
        

    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)