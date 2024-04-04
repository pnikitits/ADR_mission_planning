import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW

import os
import wandb

from src.rlglue.rl_glue import RLGlue
from src.rlglue.agent import BaseAgent
from src.agent.replay_buffer import ReplayBuffer, Transition
from src.agent.action_value_network import ActionValueNetwork
from src.environment.ADR_Environment import ADR_Environment
from src.visualisation.plot_script import plot_result
    

class Agent(BaseAgent):
    """ 
    Deep Q-Learning Agent with Experience Replay 
    """
    def __init__(self):
        self.name = 'dqn'

    def agent_init(self, agent_config):
        self.device = agent_config['device']
        self.replay_buffer = ReplayBuffer(agent_config["replay_buffer_size"],
                                          agent_config["minibatch_size"]
                                          )
        self.policy_network = ActionValueNetwork(agent_config["network_config"]).to(self.device)
        self.target_network = ActionValueNetwork(agent_config["network_config"]).to(self.device)
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
            - state (Numpy array): the state.
        Returns:
            - the action (int).
        """
        state = torch.tensor(state, device = self.device, dtype = torch.float32)
        action = self.policy_network.select_action(state, self.tau)
        # action = self.policy_network.softmax_to_greedy_action(state, self.tau)
        # action = self.policy_network.select_greedy_action(state)
        
        return action

    def agent_start(self, state):
        """
        The first method called when the experiment starts, called after
        the environment starts.
        Args:
            - state (Numpy array): the state from the
                environment's env_start function.
        Returns:
            - The first action the agent takes (int).
        """

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
            - reward (float): the reward received for taking the last action taken
            - state (Numpy array): the state from the
                environment's step based, where the agent ended up after the
                last step
        Returns:
            - The action the agent is taking (int).
        """
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.array([state])

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
            - reward (float): the reward the agent received for entering the
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
        Optimize the policy network using the experiences
        """        
        # transpose the batch from batch-array of transitions to Transition of batch-array
        batch = Transition(*zip(*experiences))

        # transform all batches into tensors
        non_final_mask = torch.tensor([not s for s in batch.terminal], device=self.device, dtype=torch.int64)


        batch_next_state_np = np.stack(batch.next_state)
        non_final_next_states = torch.from_numpy(batch_next_state_np).to(device=self.device, dtype=torch.float32).squeeze(1)

        batch_state_np = np.stack(batch.state)
        state_batch = torch.from_numpy(batch_state_np).to(device=self.device, dtype=torch.float32).squeeze(1)


        action_batch = torch.tensor(batch.action, device=self.device).unsqueeze(-1)
        reward_batch = torch.tensor(batch.reward, device=self.device, dtype=torch.float32)

        # Compute Q(s_t, a)
        state_action_values = self.policy_network(state_batch).gather(1, action_batch)

        # Use a mask to overrride the Q values of terminal states to 0
        with torch.no_grad():
            next_state_values_before_mask = self.target_network(non_final_next_states).max(1).values
            next_state_values = next_state_values_before_mask * non_final_mask

        # Compute the expected Q values (TD targets)
        expected_state_action_values = (next_state_values * self.discount) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.retain_grad()
        loss.backward()
        self.ep_loss += loss.to(torch.int32)

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_network.parameters(), 100) 
        self.optimizer.step()