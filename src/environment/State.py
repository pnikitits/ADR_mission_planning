import numpy as np
from src.environment.Strat_1 import CV , strat_1_dv
from astropy import units as u

class State:
    def __init__(self,
                 removal_step: int,
                 total_n_debris: int,
                 dv_max_per_mission,
                 dt_max_per_mission,
                 first_debris: int,
                 priority_is_on: bool,
                 can_see_priority: bool):
        
        self.removal_step = removal_step
        self.number_debris_left = total_n_debris
        self.dv_left = dv_max_per_mission
        self.dt_left = dt_max_per_mission
        self.current_removing_debris = first_debris # Index
        self.binary_flags = np.zeros(total_n_debris).tolist()
        self.binary_flags[self.current_removing_debris] = 1
        self.priority_list = np.ones(total_n_debris).tolist()
        self.priority_is_on = priority_is_on
        self.can_see_priority = can_see_priority

        # Used for normalising
        self.dt_max_per_mission = dt_max_per_mission

    def transition_function(self , action ,  cv , dt_min , priority_debris, debug = True): # Looks like it works
        
        self.removal_step += 1
        self.number_debris_left -= 1
        # self.dt_left -= action[1] # NOT dt_min ?
        self.dt_left -= dt_min.to(u.day).value
        
        #print(f"--- Taking action {action}: 'dv={cv} , dt={dt_min}")

        # print(f"{otv}")
        # print(f"{target}")

        self.dv_left -= cv.to(u.km/u.s).value
        # Update current removing debris after computing CB
        self.current_removing_debris = action[0]
        self.binary_flags[self.current_removing_debris] = 1

        # Add a higher priority to the selected debris
        if priority_debris != None:
            if self.priority_is_on:
                self.priority_list[priority_debris] = 10


    def to_list(self):
        # create a list of zeros same lenght as the priority list
        p_list = np.zeros(len(self.priority_list)).tolist()
        if self.can_see_priority: # Toggle to give access to the priority list
            p_list = self.priority_list

        #print(f"Priority list: {p_list}")
        #print(f"Binary flags: {self.binary_flags}")

        return [self.removal_step , 
                self.number_debris_left , 
                self.dv_left ,
                self.dt_left / self.dt_max_per_mission , # Normalized
                self.current_removing_debris] + self.binary_flags + p_list
        


        
