import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActionValueNetwork(nn.Module):
    """
    Action value network
    """
    def __init__(self, network_config):
        super(ActionValueNetwork, self).__init__()
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
            - The action-values (torch.tensor) : a 2D tensor of shape (batch_size, num_actions)
        """
        x = F.relu(self.l1(state))
        x = F.relu(self.l2(x))
        out = self.l3(x)

        return out.squeeze(1)

    def select_action(self, state, tau):
        """
        Softmax policy. The maximum action-value is substracted from the action-values 
        to stabilize the policy as exponentiating action values can make them very large.
        
        Args:
            - state (torch.tensor) : a 2D tensor of shape (batch_size, state_size)
            - tau : temperature argument
        Returns:
            - The action (int) : the action to take (in the range 1-300)
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
