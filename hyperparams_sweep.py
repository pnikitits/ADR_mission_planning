import torch
import wandb
import yaml
import sys
import argparse

from src.environment.ADR_Environment import ADR_Environment
from src.agent.pytorch_agent import Agent
from src.trainer.trainer import run_experiment


def run_sweeping():
    a = wandb.init()
    weights_file = None #'models/test_weights.pth'
    experiment_parameters = {"num_runs":1,
                                "num_episodes":2000,
                                "timeout":2000,
                                "gpu_use":True,
                                "track_wandb":False}
    environment_parameters = {}
    current_env = ADR_Environment
    agent_parameters = {"network_config":{"state_dim":25,
                                            "num_hidden_units":512,
                                            "num_actions":300,
                                            "weights_file":weights_file},
                            "optimizer_config":{"step_size": wandb.config.learning_rate, # working value 1e-3
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
    else:
        device = torch.device("cpu")

    agent_parameters['device'] = device
    print(device)
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
