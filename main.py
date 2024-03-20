import torch
import wandb
import yaml

from src.environment.ADR_Environment import ADR_Environment
from src.agent.pytorch_agent import Agent
from src.trainer.trainer import run_experiment


if __name__ == "__main__":
    current_env = ADR_Environment
    a = wandb.init()
    

    with open("src/config/config.yaml") as file: # change file name to use different sweep
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent_parameters = config['agent_parameters']
    experiment_parameters = config['experiment_parameters']
    environment_parameters = config['environment_parameters']

    weights_file = None #'models/test_weights.pth'
    agent_parameters['weights_file'] = weights_file


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



