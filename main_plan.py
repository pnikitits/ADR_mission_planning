import torch
import wandb
import yaml
import time

from src.environment.ADR_Environment import ADR_Environment
from src.agent.pytorch_agent import Agent
from src.trainer.trainer import run_experiment



def run(seed:int, prio:bool):
    """
    Run the experiment with the given seed and priority setting.
    
    Parameters
    ----------
    seed : int
        The seed to use for the experiment.
    prio : bool
        Whether to use priority or not.
    """
    
    start_time = time.perf_counter()

    current_env = ADR_Environment
    a = wandb.init()
    
    # Set the wandb run name
    wandb.run.name = f"seed_{seed}_prio_{prio}"
    

    with open("src/config/exhaustive_config_10.yaml") as file: # change file name to use different sweep
        config = yaml.load(file, Loader=yaml.FullLoader)

    agent_parameters = config['agent_parameters']
    agent_parameters['seed'] = seed # Manually set seed here
    
    experiment_parameters = config['experiment_parameters']
    environment_parameters = config['environment_parameters']
    environment_parameters['can_see_priority'] = prio # Manually set priority here
    
    print('env info upper: ', environment_parameters)

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


    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")


    # End the wandb run
    wandb.finish()




if __name__ == "__main__":
    
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    
    for seed in seeds:
        run(seed=seed, prio=False)
        run(seed=seed, prio=True)