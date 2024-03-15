import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from collections import deque, namedtuple
import random
from torch.optim import AdamW

import shutil
import pickle
import os
import wandb

from rl_glue import RLGlue
from agent import BaseAgent
from environment import BaseEnvironment
from ADR_Environment import ADR_Environment
from plot_script import plot_result


class DQN(nn.Module):
    """
    Action value network
    """
    def __init__(self, network_config):
        super(DQN, self).__init__()
        self.state_dim = network_config.get("state_dim")
        self.num_hidden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        self.in_weights = network_config.get("weights_file")
        self.seed = network_config.get("seed")
        self.rand_generator = np.random.RandomState(self.seed)

        self.l1 = nn.Linear(self.state_dim, self.num_hidden_units)
        self.l2 = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.l3 = nn.Linear(self.num_hidden_units, self.num_actions)

        if self.in_weights:
            self.load_state_dict(torch.load(self.in_weights))
            print("Weights loaded from ", self.in_weights)
        
    def forward(self, state):
        """
        Args:
            - state (torch.tensor) : a 2D tensor of shape (batch_size, state_size)
        Returns:
            The action-values (torch.tensor) : a 2D tensor of shape (batch_size, num_actions)
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        out = self.l3(x)

        return out.squeeze(1)

    def select_action(self, state, tau):
        """
        Forward pass + softmax policy + action sampling. The maximum action-value is substracted from the action-values 
        to stabilize the policy as exponentiating action values can make them very large.
        
        Args:
        - state (torch.tensor) : a 2D tensor of shape (batch_size, state_size)
        - tau : temperature argument
        """
        with torch.no_grad():
            preferences = self.forward(state)
        max_pref = torch.max(preferences, axis = 1).values
        reshaped_max_pref = torch.unsqueeze(max_pref, 1)
        exp_preferences = torch.exp(preferences/tau - reshaped_max_pref/tau)
        sum_of_exp_preferences = torch.sum(exp_preferences, dim = 1)
        reshaped_sum_of_exp_preferences = torch.unsqueeze(sum_of_exp_preferences, 1)
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences

        action = self.rand_generator.choice(self.num_actions, p=action_probs.detach().numpy().squeeze())
        return action

# create a tuple subclass that will be used to store transitions
Transition = namedtuple('Transition', 
                        ('state', 'action', 'reward', 'terminal', 'next_state'))

class ReplayBuffer():
    def __init__(self, size, minibatch_size):
        self.minibatch_size = minibatch_size
        self.max_size = size
        self.buffer = deque([], maxlen = self.max_size)

    def append(self, state, action, reward, terminal, next_state):
        self.buffer.append(Transition(state, action, reward, terminal, next_state))

    def sample(self):
        return random.sample(self.buffer, self.minibatch_size)
    
    def size(self):
        return len(self.buffer)
    

class Agent(BaseAgent):
    """ 
    link between replay buffer, DQN, Adam
    """
    def __init__(self):
        self.name = 'dqn'

    def agent_init(self, agent_config):
        self.replay_buffer = ReplayBuffer(agent_config["replay_buffer_size"],
                                          agent_config["minibatch_size"]
                                          )
        self.policy_network = DQN(agent_config["network_config"])
        self.target_network = DQN(agent_config["network_config"])
        self.optimizer = torch.optim.AdamW(self.policy_network.parameters(),
                              lr = agent_config["optimizer_config"]["step_size"],
                              betas = (agent_config["optimizer_config"]["beta_m"], agent_config["optimizer_config"]["beta_v"]),
                              eps = agent_config["optimizer_config"]["epsilon"],
                              amsgrad = True)
        self.num_actions = agent_config["network_config"]["num_actions"]
        self.num_replay = agent_config["num_replay_updates_per_step"]
        self.discount = agent_config["gamma"]
        self.tau = agent_config["tau"]
        self.last_state = None
        self.last_action = None
        self.sum_rewards = 0
        self.episode_steps = 0
        self.ep_loss = 0

    def policy(self, state):
        """
        Args:
            state (Numpy array): the state.
        Returns:
            the action. 
        """
        state = torch.tensor(state, device = device, dtype = torch.float32)
        action = self.policy_network.select_action(state, self.tau)
        return action

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            The first action the agent takes.
        """

        print("State in agent_start =" , state[1])

        self.ep_loss = 0
        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state[1]])

        self.last_action = self.policy(self.last_state)
        return self.last_action

    def agent_step(self, reward, state):
        """
        A step taken by the agent.
        Args:
            reward (float): the reward received for taking the last action taken
            state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            The action the agent is taking.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.array([state])
        print('***' * 50)
        print("state  = ",state)
        print('***' * 50)

        action = self.policy(state)
        self.replay_buffer.append(self.last_state , self.last_action , reward , 0 , state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                self.optimize_network(experiences)
        self.last_state = state
        self.last_action = action
        return action

    def agent_end(self, reward):
        """
        Run when the agent terminates.
        Args:
            reward (float): the reward the agent received for entering the
                terminal state.
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.zeros_like(self.last_state)
        self.replay_buffer.append(self.last_state , self.last_action , reward , 1 , state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            self.target_network.load_state_dict(self.policy_network.state_dict())
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                self.optimize_network(experiences) 

    def agent_message(self, message):
        if message == "get_sum_reward":
            return self.sum_rewards
        elif message == "get_loss":
            return self.ep_loss
        else:
            raise Exception("Unrecognised Message !")

    def optimize_network(self, experiences):
        """
        loss computation, soft update ......
        """        
        # transpose the batch from batch-array of transitions to Transition of batch-array
        batch = Transition(*zip(*experiences))

        # concatenate batch elements
        non_final_mask = torch.tensor([not s for s in batch.terminal], device=device, dtype=torch.int64)
        non_final_next_states = torch.tensor(batch.next_state, device=device, dtype=torch.float32).squeeze(1)
        state_batch = torch.tensor(batch.state, device=device, dtype=torch.float32).squeeze(1)
        action_batch = torch.tensor(batch.action, device=device).unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, device=device, dtype=torch.float32)

        # state action values = [batch_size, 1]
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.replay_buffer.minibatch_size, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_network(non_final_next_states).max(1).values 
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss() #nn.MSELoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.retain_grad()
        loss.backward()
        self.ep_loss += loss.to(torch.int32)

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100) 
        self.optimizer.step()


def run_experiment(environment , agent , environment_parameters , agent_parameters , experiment_parameters):
    """
    agent interacts with environment
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
            if track_wandb:
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
 
    subfolder = 'models/'
    model_name = input("model name : ")
    if not os.path.exists("models"):
        os.makedirs("models")
    if model_name != '':
        torch.save(rl_glue.agent.policy_network.state_dict(), subfolder + model_name +'.pth')
        print("Model saved as ", model_name)

    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists("results"):
        os.makedirs("results")
    np.save("results/sum_reward_{}".format(save_name) , agent_sum_reward)

if __name__ == "__main__":

    weights_file = None #'models/test_weights.pth'
    experiment_parameters = {"num_runs":1,
                            "num_episodes":3000,
                            "timeout":2000,
                            "gpu_use":False,
                            "track_wandb":True}
    environment_parameters = {}
    current_env = ADR_Environment
    agent_parameters = {"network_config":{"state_dim":25,
                                        "num_hidden_units":512,
                                        "num_actions":300,
                                        "weights_file":weights_file},
                        "optimizer_config":{"step_size":1e-4, # working value 1e-3 # learning rate 
                                            "beta_m":0.9,
                                            "beta_v":0.999,
                                            "epsilon":1e-8},
                        "replay_buffer_size":500000,
                        "minibatch_size":8,
                        "num_replay_updates_per_step":4,
                        "gamma":0.99,
                        "tau":0.001,
                        "seed":0
                        }
    
    # Set seed
    #seed = agent_parameters['seed']
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    #if gpu_use and torch.cuda.is_available():
        #torch.cuda.manual_seed(seed)
        #torch.cuda.manual_seed_all(seed)
        #torch.backends.cudnn.deterministic = True
        #torch.backends.cudnn.benchmark = False

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

    #plot_result(["dqn"])


    ## TO DO ##
    ## clean le code et le commenter
    ## Hyperparameter optimization
    ## Policy visualisation (+ replay buffer t-sne ?)

    # error 
    #Complete_pytorch.py:221: UserWarning: Creating a tensor from a list of numpy.ndarrays is extremely slow. Please consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor. (Triggered internally at C:\cb\pytorch_1000000000000\work\torch\csrc\utils\tensor_new.cpp:278.)
    #non_final_next_states = torch.tensor(batch.next_state, device=device, dtype=torch.float32).squeeze(1)