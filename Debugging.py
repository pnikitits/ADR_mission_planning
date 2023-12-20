from environment import BaseEnvironment
from rl_glue import RLGlue
from ADR_Environment import ADR_Environment
import DataReader

env_class = ADR_Environment

# Testing env init
env = env_class()
env.env_init()

# Testing the action space
action_key = 22
print(env.action_space[action_key])
action = env.action_space[action_key]


action_space_var = env.action_space.values()
for i, action in enumerate(action_space_var):
    
    if i % 30 == 5:
        print(i)
        reward = env.calculate_reward(action)
    