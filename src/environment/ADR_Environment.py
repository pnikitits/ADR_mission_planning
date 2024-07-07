from src.rlglue.environment import BaseEnvironment
import numpy as np
from src.environment.State import State
from src.environment.InPlaneEquations import *
from src.environment.Debris import Debris
from src.environment.Strat_1 import strat_1_dv#DT_required, CV
import random
from src.simulator.Simulator import Simulator

from astropy import units as u


class ADR_Environment(BaseEnvironment):
    def init(self):
        self.name = "ADR"


    def env_init(self, first_debris=3, env_info={}):
        
        # we only set the environment parameters at the first episode
        if env_info != {}:
            self.total_n_debris = env_info["total_n_debris"] # TODO gets len debris after datareader
            self.dv_max_per_mission = env_info['dv_max_per_mission'] # * u.km / u.s
            self.dt_max_per_mission = env_info['dt_max_per_mission'] # * u.day
            self.dt_max_per_transfer = env_info['dt_max_per_transfer'] # * u.day       
            self.priority_is_on = env_info['priority_is_on']   # Boolean
            self.time_based_action = env_info['time_based_action'] # Boolean
            self.random_first_debris = env_info['random_first_debris']
            self.refuel_station_is_on = env_info['refuel_station_is_on']

        # Debugging
        self.debug = True
        self.debug_list = [0, 0, 0, 0]

        self.fuel_uses_in_episode = [] # to log the fuel use
        self.time_uses_in_episode = []
        
        # Init starting debris
        if self.random_first_debris:
            self.first_debris = random.randint(0, self.total_n_debris-1)
        else:
            self.first_debris = first_debris
            print('first debris: ', self.first_debris) if self.debug else None

        self.simulator = Simulator(starting_index=self.first_debris , n_debris=self.total_n_debris)

        self.action_is_legal = False
        
        # Use the correct dictionary for the action space
        if self.time_based_action:
            self.action_space = self.action_dict()
        else: 
            self.action_space = self.no_time_action_dict()
        
        self.action_space_len = len(self.action_space)

        
        # Initial values
        self.state = State(removal_step = 0 ,
                           total_n_debris = self.total_n_debris ,
                           dv_max_per_mission = self.dv_max_per_mission ,
                           dt_max_per_mission = self.dt_max_per_mission ,
                           first_debris = self.first_debris,
                           priority_is_on = self.priority_is_on,
                        #  refuel_station_indices=self.init_refuel_indices(),
                           refuel_station_indices=[1],
                           refuel_station_is_on = self.refuel_station_is_on,
                           refuel_amount=1.0,
                            )
        

        observation = self.env_observe_state()
        self.last_observation = observation

        
        return observation


    def env_observe_state(self):
        return self.state.to_list() #[self.removal_step , self.number_debris_left , self.dv_left , self.dt_left , self.current_removing_debris] + self.binary_flags


    def is_legal(self , action , cv , dt_min):
        # input is state before transition

        next_debris_index , dt_action = action
        """
        Min time
        check if action is possible:
        tr1 = True
        """
        tr1 = False
        if dt_action * u.day > dt_min:
            tr1 = True

        """
        Max time
        check if next_state_t_left > 0:
        tr2 = True
        """
        tr2 = False
        if (self.state.dt_left - dt_action) > 0:
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
        if (self.state.dv_left * (u.km/u.s) - cv) > 0:
            tr4 = True
        

        self.debug_list = [tr1, tr2, tr3, tr4]

        if (tr1 and tr2 and tr3 and tr4):
            self.fuel_uses_in_episode.append(cv.to(u.m/u.s).value)
            self.time_uses_in_episode.append(dt_min.to(u.s).value)

        return (tr1 and tr2 and tr3 and tr4)

    

    def env_start(self, first_debris=0):
        print("\nENV START\n") if self.debug else None
        reward = 0.0
        is_terminal = False

        # values update

        observation = self.env_init(first_debris)
        # print(observation)

        print('\n ----- Starting Episode ---- \n') if self.debug else None

        return (reward, observation, is_terminal)

    def update_debris_pos(self, action):
        # Iterate through debris list to update positions
        # TODO: REMOVE THIS
        for debris in self.debris_list:
            debris.update(action[1]*u.day)
        
    def compute_reward(self, action):
        # Calculate reward using the priority list
        action_target = action[0]
        reward = self.state.priority_list[action_target]

        # Set reward to 0 if the action is not legal
        if not self.action_is_legal:
            reward = 0

        # Set reward to 0 if at refuel station
        if self.state.refuel_station_binary_flags[action_target] == 1:
            reward = 0
        
        return reward


    def env_step(self, action_key):

        print("\n -----  ENV STEP ----- \n") if self.debug else None

        print('action_key: ', action_key) if self.debug else None

        # Convert action key from NN into action (next_debris_norad_id , dt_given)
        action = self.action_space[action_key]

        print('converted action: ', action) if self.debug else None

        # print(f"Action: {action} , otv at: {self.state.current_removing_debris}") # If the action is not legal by binary flags, the propagation does NOT work
        # print(f"Next binary flag: {self.state.binary_flags[action[0]]}")
        if self.state.binary_flags[action[0]] == 1:
            print('illegal binary flag') if self.debug else None
            return (0 , self.state.to_list() , True)
            

        # Use the simulator to compute the maneuvre fuel and time and propagate
        cv , dt_min = self.simulator.simulate_action(action)

        # DEBUG: Check that the otv has moved to the correct derbis
        target_debris = self.simulator.debris_list[action[0]].poliastro_orbit
        otv = self.simulator.otv_orbit
        diff = otv.r - target_debris.r
        if np.max(diff) > 0.1 * u.km:
            print('distance between otv and target debris: ', otv.r - target_debris.r)
            print('time differences: ', (otv.epoch - target_debris.epoch))

        self.action_is_legal = self.is_legal(action , cv , dt_min)
        if not self.action_is_legal and self.debug:
            print('max fuel used')

        # Get reward based on action
        reward = self.compute_reward(action)

        # Check if terminal
        is_terminal = not self.action_is_legal

        # Reset the priority list
        self.state.priority_list = np.ones(self.total_n_debris).tolist()

        # Update the priority list on a random basis
        priority_debris = self.get_priority()

        # Update the state
        self.state.transition_function(action=action , cv=cv , dt_min=dt_min, priority_debris=priority_debris)

        #if self.debug:
        #    print(' -------- New state ------')
        #    print(self.state.to_list())
        #    print(' --- BINARY FLAGS -- ')
        #    print(self.state.binary_flags)


        return (reward, self.state.to_list(), is_terminal)



    def env_cleanup(self):
        self.env_init()

    
    def action_dict(self):
        # Used in the case where the agent can select the dt per maneouvre
        dict = {}
        i = 0
        for debris in range(self.total_n_debris):
            for dt in range(self.dt_max_per_transfer):
                dict[i] = (debris , dt)
                i += 1
        return dict
    
    def no_time_action_dict(self):
        # Use the in the case where the agent can only select the debris
        dict = {}
        i = 0
        for debris in range(self.total_n_debris):
            # Still have a condition on the max time per transfer
            dict[i] = (debris , self.dt_max_per_transfer)
            i += 1
        return dict
    
    def get_priority(self):
        '''
            Returns a random debris index to set as priority
            Taken from the available debris that have not been removed yet
        '''
        priority_debris = None

        # Get the list of indices where the binary flag is 0
        available_debris = [i for i, flag in enumerate(self.state.binary_flags) if flag == 0 and self.state.refuel_station_binary_flags[i]==0]

        if random.random() < 0.3:
            # Randomly select a debris from the available list
            priority_debris = random.choice(available_debris)

            # priority_debris = random.randint(0, self.total_n_debris-1)
        return priority_debris
    
    def init_refuel_indices(self):
        # Return a list of indices where the refuel stations are located
        nb_refuel_stations = self.total_n_debris // 6
        refuel_indices = random.sample(range(self.total_n_debris), nb_refuel_stations)
        print('Refuel indices: ', refuel_indices) if self.debug else None
        return refuel_indices
        

    def init_debris(self):
        pass

    def init_debris_from_data(self):
        pass

        
    def get_term_reason(self):
        # Return 1 if terminal state caused by this condition
        impossible_dt = hash(not self.debug_list[0])
        time_limit = hash(not self.debug_list[1])
        impossible_binary_flag = hash(not self.debug_list[2])
        fuel_limit = hash(not self.debug_list[3])

        return fuel_limit, time_limit, impossible_dt, impossible_binary_flag
    
    def get_fuel_use_average(self):
        if len(self.fuel_uses_in_episode) > 0:
            return np.mean(self.fuel_uses_in_episode)
        else:
            return 0
        
    def get_time_use_average(self):
        if len(self.time_uses_in_episode) > 0:
            return np.mean(self.time_uses_in_episode)
        else:
            return 0