import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm 
import random

import wandb
import yaml
import argparse
import sys


from src.environment.ADR_Environment import ADR_Environment
from src.rlglue.rl_glue import RLGlue
from src.agent.pytorch_agent import Agent, Transition, ReplayBuffer, ActionValueNetwork


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
        agent_info["seed"] = 0 #run
        agent_info["network_config"]["seed"] = 0 #run
        env_info["seed"] = 0 #run
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



def run_sweeping():
    a = wandb.init()
    weights_file = None #'models/test_weights.pth'
    experiment_parameters = {"num_runs":1,
                                "num_episodes":50,
                                "timeout":30,
                                "gpu_use":False,
                                "track_wandb":False}
    environment_parameters = {}
    current_env = ADR_Environment
    agent_parameters = {"network_config":{"state_dim":25,
                                            "num_hidden_units":512,
                                            "num_actions":300,
                                            "weights_file":weights_file},
                            "optimizer_config":{"step_size": wandb.config.learning_rate, 
                                                "beta_m":0.9,
                                                "beta_v":0.999,
                                                "epsilon":1e-8},
                            "replay_buffer_size":wandb.config.replay_buffer_size,
                            "minibatch_size":wandb.config.minibatch_size,
                            "num_replay_updates_per_step": wandb.config.replay_updates_per_step,
                            "gamma":wandb.config.gamma,
                            "tau":wandb.config.tau,
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
    
    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)

def main(args):
    parser = argparse.ArgumentParser()

    # Argument parser
    parser.add_argument(
        "-method",
        dest="method",
        type=str,
        default="bayes",
        required=False,
    )

    parser.add_argument(
        "-nb_sweeps",
        dest="nb_sweeps",
        type=int,
        default=20,
        required=False,
    )

    args = parser.parse_args()
    if args.method == "bayes":
        with open("./src/config/bayes_sweep.yaml") as file:
            sweep_configuration = yaml.load(file, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="HPO-ADR")
        wandb.agent(sweep_id, function=run_sweeping, count = args.nb_sweeps)
    elif args.method == "grid":
        with open("./src/config/grid_sweep.yaml") as file:
            sweep_configuration = yaml.load(file, Loader=yaml.FullLoader)
        sweep_id = wandb.sweep(sweep=sweep_configuration, project="HPO-ADR")
        wandb.agent(sweep_id, function=run_sweeping)
    else:
        raise NotImplementedError("Agent not implemented yet.")



if __name__ == "__main__":
    main(sys.argv[1:])