import torch
import wandb
import yaml
import time

from src.environment.ADR_Environment import ADR_Environment
from src.agent.pytorch_agent import Agent
from src.trainer.trainer import run_experiment


if __name__ == "__main__":
    start_time = time.perf_counter()

    current_env = ADR_Environment
    

    with open("src/config/config.yaml") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent_parameters = config['agent_parameters']
    experiment_parameters = config['experiment_parameters']
    environment_parameters = config['environment_parameters']
    print('env info upper: ', environment_parameters )

    a = wandb.init() if experiment_parameters['track_wandb'] else None

    # Set device
    gpu_use = experiment_parameters['gpu_use']
    
    if gpu_use and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    agent_parameters['device'] = device
    print(device)
    current_agent = Agent

    # track only if wandb is enabled
    #a = wandb.init() if experiment_parameters['track_wandb'] else None
    
    run_experiment(current_env, current_agent, environment_parameters, agent_parameters, experiment_parameters)


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
