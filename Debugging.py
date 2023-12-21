from environment import BaseEnvironment
from rl_glue import RLGlue
from ADR_Environment import ADR_Environment
import DataReader

env_class = ADR_Environment

# Testing env init
env = env_class()
env.env_init()

# Testing the action space
action_key = 3*30+0
print(env.action_space[action_key])
action = env.action_space[action_key]


# action_space_var = env.action_space.values()
# for i, action in enumerate(action_space_var):
    
#     if i % 30 == 5:
#         print(i)
#         reward = env.calculate_reward(action)

# Check env step
reward, state, is_term = env.env_step(action_key)
print('reward: ', reward)
print('state: ', state)
print('is terminal: ', is_term)