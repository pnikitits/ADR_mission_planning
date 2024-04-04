import torch
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np


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
        torch.manual_seed(self.seed)
        #self.rand_generator = np.random.RandomState(self.seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.l1 = nn.Linear(self.state_dim, self.num_hidden_units)
        self.l2 = nn.Linear(self.num_hidden_units, self.num_hidden_units)
        self.l3 = nn.Linear(self.num_hidden_units, self.num_actions)

        if self.in_weights:
            self.load_state_dict(torch.load(self.in_weights , map_location=self.device))
            print("Weights loaded from ", self.in_weights)

        self.action_counter = 0
        
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

        # with torch.no_grad():
        #     preferences = self.forward(state)
        # action_probs = F.softmax(preferences / tau, dim=1)
        # action = torch.multinomial(action_probs, num_samples=1).item()
        # return action
        with torch.no_grad():
            preferences = self.forward(state)
        max_pref = torch.max(preferences, axis = 1).values
        reshaped_max_pref = torch.unsqueeze(max_pref, 1)
        exp_preferences = torch.exp(preferences/tau - reshaped_max_pref/tau)
        sum_of_exp_preferences = torch.sum(exp_preferences, dim = 1)
        reshaped_sum_of_exp_preferences = torch.unsqueeze(sum_of_exp_preferences, 1)
        action_probs = exp_preferences / reshaped_sum_of_exp_preferences


        action = torch.multinomial(action_probs, num_samples=1) # do this on the GPU
        action = action.item()
        # # action = self.rand_generator.choice(self.num_actions, p=action_probs.detach().numpy().squeeze())
        return action
    
    def select_greedy_action(self, state):
        """
            Pure greedy policy
        
        Args:
            - state (torch.tensor) : a 2D tensor of shape (batch_size, state_size)
        Returns:
            - The action (int) : the action to take (in the range 1-300)
        """

        with torch.no_grad():
            preferences = self.forward(state)
        
        print('state:', state)
        print('preferences:', preferences)
        action = torch.argmax(preferences, axis = 1).item()

        return action

    def softmax_to_greedy_action(self, state, tau, greedy_switch=1000):
        '''
            Switches from softmax to greedy policy after a certain number of episodes

            Args:
                - state (torch.tensor) : a 2D tensor of shape (batch_size, state_size)
                - tau : temperature argument
                - ep_count : episode count
                - greedy_switch : episode number to switch to greedy policy
            Returns:
                - The action (int) : the action to take (in the range 1-300)
        '''

        self.action_counter += 1

        if self.action_counter < greedy_switch:
            action = self.select_action(state, tau)
        else:
            action = self.select_greedy_action(state)
            print("Switched to greedy policy")

        return action