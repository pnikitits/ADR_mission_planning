from environment import BaseEnvironment
import numpy as np
from State import State
from PhysicsEquations import *
from Debris import Debris


class ADR_Environment(BaseEnvironment):
    def init(self):
        self.name = "ADR"
        

    def env_init(self , env_info={}):
        self.total_n_debris = 10
        self.dv_max = 100
        self.dt_max_per_mission = 365
        self.dt_max_per_transfer = 30
        self.first_debris = 4

        self.debris_list = []
        
        # Init randomly for testing
        self.init_random_debris()

        self.action_space = self.action_dict()
        self.action_space_len = len(self.action_space)

        # Initial values
        self.state = State(0 ,
                           self.total_n_debris ,
                           self.dv_max ,
                           self.dt_max_per_transfer ,
                           self.first_debris)


    def env_observe_state(self):
        return [self.removal_step , self.number_debris_left , self.dv_left , self.dt_left , self.current_removing_debris] + self.binary_flags


    def calculate_reward(self , action):
        if self.is_legal(action) :
            return 1
        else:
            return 0
        


    def is_legal(self , action):
        # input is state before transition
        
        next_debris_index , dt = action

        otv = self.debris_list[self.state.current_removing_debris]
        target = self.debris_list[next_debris_index]

        """
        Min time
        check if dt > hohmann_t + phase_t:
        tr1 = True
        """
        tr1 = False
        if dt > (hohmann_time(otv.a , target.a)) + phase_time(otv , target):
            tr1 = True

        """
        Max time
        check if next_state_t_left > 0:
        tr2 = True
        """
        tr2 = False
        if (self.state.dt_left - dt) > 0:
            tr2 = True

        """
        if debris is available (binary flag == 0)
        tr3 = True
        """
        tr3 = False
        if self.state.binary_flags[next_debris_index] == 0:
            tr3 = True

        """
        Max dv (fuel)
        check if next_state_dv_left > 0
        tr4 = True
        """
        tr4 = False
        if (self.state.dv_left - hohmann_dv(otv.a , target.a)) > 0:
            tr4 = True

        return (tr1 and tr2 and tr3 and tr4)



    def is_terminal(self , action):
        if not self.is_legal(action):
            return True
        return False

    

    def env_start(self):
        pass

    def update_debris_pos(self, action):
        # Iterate through debris list to update positions
        for debris in self.debris_list:
            debris.update(action[1])


    def env_step(self, action_key):

        # Convert action key from NN into action
        action = self.action_space[action_key]
        
        # Get reward based on action
        reward = self.calculate_reward(action)

        # Propagate debris positions
        self.update_debris_pos(self, action)

        # Check if terminal
        is_terminal = self.is_terminal(self.state)

        # Update the state
        self.state.transition_function(self, action)

        return (reward, self.state, is_terminal)



    def env_cleanup(self):
        self.env_init()

    
    def action_dict(self):
        # return dict = {key:tuple(debris index , dt)}
        dict = {}
        i = 0
        for debris in range(self.total_n_debris):
            for dt in range(self.dt_max_per_transfer):
                dict[i] = (debris , dt)
                i += 1
        return dict
    

    def init_debris(self):
        pass

    def init_debris_from_data(self):
        pass

    def init_random_debris(self):
        """Generate random debris"""

        np.random.seed(42)

        n = 10
        min_a = 200+6371
        max_a = 600+6371
        min_mean_anomaly = 0
        max_mean_anomaly = 360

        output = []
        for _ in range(n):
            debris = Debris(norad=None,
                            inclination=None,
                            raan=None,
                            eccentricity=0,
                            arg_perigee=None,
                            mean_anomaly=np.random.uniform(min_mean_anomaly, max_mean_anomaly),
                            a=np.random.uniform(min_a, max_a),
                            rcs=None)
            output.append(debris)

        self.debris_list = output
        