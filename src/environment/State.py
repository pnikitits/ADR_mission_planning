import numpy as np
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
        """
        Initialize the state of the environment.
        
        Parameters
        ----------
        removal_step : int
            The current step of the removal process.
        total_n_debris : int
            The total number of debris in the environment.
        dv_max_per_mission : float
            The fuel budget for the mission.
        dt_max_per_mission : float
            The time budget for the mission.
        first_debris : int
            The index of the first debris to remove.
        priority_is_on : bool
            Whether to use priority or not (higher reward for collecting priority debris).
        can_see_priority : bool
            Whether the agent can see the priority list or not.
        """
        
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


    def transition_function(self, action, cv, dt_min, priority_debris, verbose=False): # Looks like it works
        """
        Transition function for the environment.
        Increments the removal step, decreases the number of debris left, and updates the fuel and time budgets.
        """
        
        self.removal_step += 1
        self.number_debris_left -= 1
        self.dt_left -= dt_min.to(u.day).value
        
        print(f"--- Taking action {action}: 'dv={cv} , dt={dt_min}") if verbose else None

        self.dv_left -= cv.to(u.km/u.s).value
        # Update current removing debris after computing CB
        self.current_removing_debris = action[0]
        self.binary_flags[self.current_removing_debris] = 1

        # Add a higher priority to the selected debris
        if priority_debris != None:
            if self.priority_is_on:
                self.priority_list[priority_debris] = 10


    def to_list(self, verbose=False):
        """
        Make a list of the state variables.
        """
        # Create a list of zeros same lenght as the priority list
        p_list = np.zeros(len(self.priority_list)).tolist()
        
        if self.can_see_priority: # Toggle to give access to the priority list
            p_list = self.priority_list

        print(f"Priority list: {p_list}") if verbose else None
        print(f"Binary flags: {self.binary_flags}") if verbose else None

        return [self.removal_step , 
                self.number_debris_left , 
                self.dv_left ,
                self.dt_left / self.dt_max_per_mission , # Normalized
                self.current_removing_debris] + self.binary_flags + p_list