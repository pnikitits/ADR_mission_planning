import numpy as np
import matplotlib.pyplot as plt
from rlglue.rl_glue import RLGlue
from rlglue.environment import BaseEnvironment

# ARD Envinonment
from src.environment.ADR_Environment import ADR_Environment

from rlglue.agent import BaseAgent
from collections import deque
from copy import deepcopy
from tqdm import tqdm
import os
import shutil
from plot_script import plot_result

import pickle
import wandb 

from astropy import units as u



# Part 1
class ActionValueNetwork:
    def __init__(self , network_config):
        self.state_dim = network_config.get("state_dim")
        self.num_hudden_units = network_config.get("num_hidden_units")
        self.num_actions = network_config.get("num_actions")
        self.in_weights = network_config.get("weights_file")
        self.rand_generator = np.random.RandomState(network_config.get("seed"))
        self.layer_sizes = [self.state_dim , self.num_hudden_units , self.num_actions]

        if self.in_weights == None:
            print("No input weights, start from scratch")
            self.weights = [dict() for i in range(0 , len(self.layer_sizes) - 1)]
            for i in range(0 , len(self.layer_sizes) - 1):
                self.weights[i]['W'] = self.init_saxes(self.layer_sizes[i] , self.layer_sizes[i+1])
                self.weights[i]["b"] = np.zeros((1 , self.layer_sizes[i+1]))
            #print(self.weights)
        else:
            print("self.weights set to input weights")
            self.weights = deepcopy(load_weights(self.in_weights))
            #print(self.weights)


    def get_action_values(self , s):
        W0 , b0 = self.weights[0]["W"] , self.weights[0]["b"]
        #print("S in get_action_value =" , s)
        # print("Shapes - W0:", W0.shape, "s:", s.shape, "b0:", b0.shape)
        

        psi = np.dot(s , W0) + b0
        x = np.maximum(psi , 0)
        W1 , b1 = self.weights[1]["W"] , self.weights[1]["b"]
        #print("Shapes - x:", x.shape, "W1:", W1.shape, "b1:", b1.shape)
        q_vals = np.dot(x , W1) + b1
        return q_vals

    def get_TD_update(self , s , delta_mat):
        #print("S in get_TD_update =" , s)
        W0 , b0 = self.weights[0]["W"] , self.weights[0]["b"]
        W1 , b1 = self.weights[1]["W"] , self.weights[1]["b"]
        psi = np.dot(s , W0) + b0
        x = np.maximum(psi , 0)
        dx = (psi > 0).astype(float)
        td_update = [dict() for i in range(len(self.weights))]
        v = delta_mat
        td_update[1]["W"] = np.dot(x.T , v) * 1. / s.shape[0]
        td_update[1]["b"] = np.sum(v , axis=0 , keepdims=True) * 1. / s.shape[0]
        v = np.dot(v , W1.T) * dx
        td_update[0]["W"] = np.dot(s.T , v) * 1. / s.shape[0]
        td_update[0]["b"] = np.sum(v , axis=0 , keepdims=True) * 1. / s.shape[0]
        return td_update
    
    def init_saxes(self , rows , cols):
        #print("ROWS" , rows)
        #print("COLS" , cols)
        tensor = self.rand_generator.normal(0 , 1 , (rows , cols))
        if rows < cols:
            tensor = tensor.T
        tensor , r = np.linalg.qr(tensor)
        d = np.diag(r , 0)
        ph = np.sign(d)
        tensor *= ph
        if rows < cols:
            tensor = tensor.T
        return tensor
    
    def get_weights(self):
        return deepcopy(self.weights)
    
    def set_weights(self , weights):
        self.weights = deepcopy(weights)


# Part 2
class Adam():
    def __init__(self , layer_sizes , optimizer_info):
        self.layer_sizes = layer_sizes
        self.step_size = optimizer_info.get("step_size")
        self.beta_m = optimizer_info.get("beta_m")
        self.beta_v = optimizer_info.get("beta_v")
        self.epsilon = optimizer_info.get("epsilon")
        self.m = [dict() for i in range(1 , len(self.layer_sizes))]
        self.v = [dict() for i in range(1 , len(self.layer_sizes))]
        for i in range(0 , len(self.layer_sizes)-1):
            self.m[i]["W"] = np.zeros((self.layer_sizes[i] , self.layer_sizes[i+1]))
            self.m[i]["b"] = np.zeros((1 , self.layer_sizes[i+1]))
            self.v[i]["W"] = np.zeros((self.layer_sizes[i] , self.layer_sizes[i+1]))
            self.v[i]["b"] = np.zeros((1 , self.layer_sizes[i+1]))
        self.beta_m_product = self.beta_m
        self.beta_v_product = self.beta_v

    def update_weights(self , weights , td_errors_times_gradients):
        for i in range(len(weights)):
            for param in weights[i].keys():
                self.m[i][param] = self.beta_m * self.m[i][param] + (1 - self.beta_m) * td_errors_times_gradients[i][param]
                self.v[i][param] = self.beta_v * self.v[i][param] + (1 - self.beta_v) * td_errors_times_gradients[i][param] ** 2
                m_hat = self.m[i][param] / (1 - self.beta_m_product)
                v_hat = self.v[i][param] / (1 - self.beta_v_product)
                weight_update = self.step_size * m_hat / (np.sqrt(v_hat) + self.epsilon)
                weights[i][param] = weights[i][param] + weight_update
        self.beta_m_product *= self.beta_m
        self.beta_v_product *= self.beta_v
        return weights
    

# Part 3
class ReplayBuffer:
    def __init__(self , size , minibatch_size , seed):
        self.buffer = []
        self.minibatch_size = minibatch_size
        self.rand_generator = np.random.RandomState(seed)
        self.max_size = size

    def append(self , state , action , reward , terminal , next_state):
        if len(self.buffer) == self.max_size:
            del self.buffer[0]
        self.buffer.append([state , action , reward , terminal , next_state])

    def sample(self):
        idxs = self.rand_generator.choice(np.arange(len(self.buffer)) , size=self.minibatch_size)
        return [self.buffer[idx] for idx in idxs]
    
    def size(self):
        return len(self.buffer)
    

# Part 4
def softmax(action_values , tau=1.0):
    preferences = action_values / tau
    max_preferences = np.max(preferences , axis=1)
    reshaped_max_preferences = max_preferences.reshape((-1 , 1))
    exp_preferences = np.exp(preferences - reshaped_max_preferences)
    sum_of_exp_preferences = np.sum(exp_preferences , axis=1)
    reshaped_sum_of_exp_preferences = sum_of_exp_preferences.reshape((-1 , 1))
    action_probs = exp_preferences / reshaped_sum_of_exp_preferences
    action_probs = action_probs.squeeze()
    return action_probs

# Part 5
def get_td_error(states , next_states , actions , rewards , discount , terminals , network , current_q , tau):
    q_next_mat = current_q.get_action_values(next_states)
    probs_mat = softmax(q_next_mat , tau)
    v_next_vec = np.sum(q_next_mat * probs_mat , axis=1) * (1-terminals)
    target_vec = rewards + discount * v_next_vec
    q_mat = network.get_action_values(states)
    batch_indices = np.arange(q_mat.shape[0])
    q_vec = q_mat[batch_indices , actions]
    delta_vec = target_vec - q_vec
    return delta_vec
    

def optimize_network(experiences , discount , optimizer , network , current_q , tau):
    states , actions , rewards , terminals , next_states = map(list , zip(*experiences))
    states = np.concatenate(states)
    next_states = np.concatenate(next_states)
    rewards = np.array(rewards)
    terminals = np.array(terminals)
    batch_size = states.shape[0]
    delta_vec = get_td_error(states , next_states , actions , rewards , discount , terminals , network , current_q , tau)
    batch_indices = np.arange(batch_size)
    delta_mat = np.zeros((batch_size , network.num_actions))
    delta_mat[batch_indices , actions] = delta_vec
    td_update = network.get_TD_update(states , delta_mat)
    weights = optimizer.update_weights(network.get_weights() , td_update)
    network.set_weights(weights)



class Agent(BaseAgent):
    def __init__(self):
        self.name = "expected_sarsa_agent"

    def agent_init(self , agent_config):
        self.replay_buffer = ReplayBuffer(agent_config["replay_buffer_size"],
                                          agent_config["minibatch_size"],
                                          agent_config.get("seed"))
        self.network = ActionValueNetwork(agent_config["network_config"])
        self.optimizer = Adam(self.network.layer_sizes , agent_config["optimizer_config"])
        self.num_actions = agent_config["network_config"]["num_actions"]
        self.num_replay = agent_config["num_replay_updates_per_step"]
        self.discount = agent_config["gamma"]
        self.tau = agent_config["tau"]
        self.rand_generator = np.random.RandomState(agent_config.get("seed"))
        self.last_state = None
        self.last_action = None
        self.sum_rewards = 0
        self.episode_steps = 0

    def policy(self , state):
        action_values = self.network.get_action_values(state)
        probs_batch = softmax(action_values , self.tau)
        action = self.rand_generator.choice(self.num_actions , p=probs_batch.squeeze())
        return action
    
    
    def agent_start(self , state):
        #print("State in agent_start =" , state[1])

        self.sum_rewards = 0
        self.episode_steps = 0
        self.last_state = np.array([state[1]])

        # print("AFTER MODIF =" , self.last_state)

        self.last_action = self.policy(self.last_state)
        return self.last_action
    
    
    def agent_step(self, reward, state):
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.array([state])
        action = self.policy(state)
        self.replay_buffer.append(self.last_state , self.last_action , reward , 0 , state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                optimize_network(experiences , self.discount , self.optimizer , self.network , current_q , self.tau)
        self.last_state = state
        self.last_action = action
        return action
    
    def agent_end(self , reward):
        self.sum_rewards += reward
        self.episode_steps += 1
        state = np.zeros_like(self.last_state)
        self.replay_buffer.append(self.last_state , self.last_action , reward , 1 , state)
        if self.replay_buffer.size() > self.replay_buffer.minibatch_size:
            current_q = deepcopy(self.network)
            for _ in range(self.num_replay):
                experiences = self.replay_buffer.sample()
                optimize_network(experiences , self.discount , self.optimizer , self.network , current_q , self.tau)

    def agent_message(self , message):
        if message == "get_sum_reward":
            return self.sum_rewards
        else:
            raise Exception("Unrecognised Message !")
        

# Part 6
def run_experiment(environment , agent , environment_parameters , agent_parameters , experiment_parameters):
    

    rl_glue = RLGlue(environment , agent)
    agnet_sum_reward = np.zeros((experiment_parameters["num_runs"],
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
            fuel_limit, time_limit, impossible_dt, impossible_binary_flag = rl_glue.environment.get_term_reason()
            avg_fuel_used = rl_glue.environment.get_fuel_use_average()
            avg_time_used = rl_glue.environment.get_time_use_average()
            # wand logging
            if track_wandb:
                wandb.log({
                    "episode reward": episode_reward ,
                    "fuel limit": fuel_limit,
                    "time limit": time_limit,
                    "impossible_dt": impossible_dt,
                    "impossible_binary_flag": impossible_binary_flag,
                    "average fuel used":avg_fuel_used,
                    "average time used":avg_time_used
                })

            

            

    save_path = input("Save path name : ")
    save_weights(path= save_path, data=rl_glue.agent.network.weights)

    save_name = "{}".format(rl_glue.agent.name)
    if not os.path.exists("results"):
        os.makedirs("results")
    np.save("results/sum_reward_{}".format(save_name) , agnet_sum_reward)
    shutil.make_archive("results" , "zip" , "results")
    


def save_weights(data , path):
    with open(path+".pickle", 'wb') as file:
        pickle.dump(data, file)
    print("Saved data at " , path)

    
def load_weights(path):
    with open(path+".pickle", 'rb') as file:
        loaded_data = pickle.load(file)
        print("Loaded data at " , path)
        return loaded_data




if __name__ == "__main__":
    #prompt1 = input("Start from scratch ? [y/n] :")

    weight_file = None
    #if prompt1 == "n":
    #    weight_file = input("input file path: ")
    

    experiment_parameters = {"num_runs":1,
                             "num_episodes":5000,
                             "timeout":2000}
    environment_parameters = {}
    current_env = ADR_Environment
    agent_parameters = {"network_config":{"state_dim":25,
                                          "num_hidden_units":512,
                                          "num_actions":300,
                                          "weights_file":weight_file},
                        "optimizer_config":{"step_size":1e-3,
                                            "beta_m":0.9,
                                            "beta_v":0.999,
                                            "epsilon":1e-8},
                        "replay_buffer_size":500000,
                        "minibatch_size":8,
                        "num_replay_updates_per_step":4,
                        "gamma":0.99,
                        "tau":0.01}
    
    
    current_agent = Agent

    # Setup wandb
    global track_wandb
    track_wandb = True
    if track_wandb:
        wandb.login()
        wandb.init(
            project="ADR Mission Planning",
            config = agent_parameters
        )

    run_experiment(current_env , current_agent , environment_parameters , agent_parameters , experiment_parameters)

    plot_result(["expected_sarsa_agent"])


    
    