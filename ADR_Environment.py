from environment import BaseEnvironment
import numpy as np
from State import State


class ADR_Environment(BaseEnvironment):
    def init(self):
        self.name = "ADR"
        

    def env_init(self , env_info={}):
        self.total_n_debris = 10
        self.dv_max = 100
        self.dt_max_per_mission = 365
        self.dt_max_per_transfer = 30
        self.fisrt_debris = 4

        self.action_space = self.action_dict()
        self.action_space_len = len(self.action_space)

        # Initial values
        self.state = State(0 ,
                           self.total_n_debris ,
                           self.dv_max ,
                           self.dt_max_per_transfer ,
                           self.fisrt_debris)



    def env_observe_state(self):
        return [self.removal_step , self.number_debris_left , self.dv_left , self.dt_left , self.current_removing_debris] + self.binary_flags


    def calculate_reward(self , state , action , next_state):
        if self.is_legal(action , next_state) :
            return 1
        else:
            return 0
        


    def is_legal(self , action , state):
        d , dt = action

        # Min time
        # check if dt > hohmann_t + phase_t:
        # tr1 = True


        # Max time
        # check if next_state_t_left > 0:
        # tr2 = True


        # if debris is available (binary flag == 0)
        # tr3 = True

        # Max dv (fuel)
        # check if next_state_dv_left > 0
        # tr4 = True


        return (tr1 and tr2 and tr3 and tr4)


    def is_terminal(self , state , action):
        if not self.is_legal(action , state):
            return True
        return False

    

    def transition_function(self , action , state):


        # Calculate next state
        return # next state
    

    def env_start(self):
        pass


    def env_step(self, action):
        self.transition_function(action)

        # state update
        next_state = self.state.transition_function(action)

        next_state = self.env_observe_state()
        reward = self.calculate_reward(self.last_observation, action, next_state)
        is_terminal = self.is_terminal(next_state)
        self.last_observation = next_state
        return (reward, next_state, is_terminal)

    def env_cleanup(self):
        self.env_init()

    
    def action_dict(self):
        # return dict = {key:tuple(debris , dt)}
        dict = {}
        i = 0
        for debris in range(self.total_n_debris):
            for dt in range(self.dt_max_per_transfer):
                dict[i] = (debris , dt)
                i += 1
        return dict