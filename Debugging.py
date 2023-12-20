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

# Test update debris pos
# print(env.debris_list[9].mean_anomaly)
# print(env.debris_list[0].angular_velocity)
# env.update_debris_pos(action)
# print(env.debris_list[0].mean_anomaly)

# Test datareader
# debris_list = DataReader.make_Iridium_debris()
# print(debris_list[0].a)
# print(debris_list[0].mean_anomaly)
# print(debris_list[0].angular_velocity)

# Test state transition
# print(env.state.removal_step)
# env.state.transition_function(env, action)
# print('after')
# print(env.state.removal_step)

action_space_var = env.action_space.values()
for i, action in enumerate(action_space_var):
    reward = env.calculate_reward(action)
    # if reward == 1:
        # print('ooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooooo')
        # print(f'pour l\'action {i} : {action}, reward = {reward}')