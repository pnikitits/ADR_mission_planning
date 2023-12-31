import numpy as np
from PhysicsEquations import Cv

class State:
    def __init__(self , removal_step , total_n_debris , dv_max_per_mission , dt_max_per_mission , first_debris):
        self.removal_step = removal_step
        self.number_debris_left = total_n_debris
        self.dv_left = dv_max_per_mission
        self.dt_left = dt_max_per_mission
        self.current_removing_debris = first_debris # Index
        self.binary_flags = np.zeros(total_n_debris).tolist()
        self.binary_flags[self.current_removing_debris] = 1

    def transition_function(self , env, action, debug = True): # Looks like it works
        
        print(' --- Taking action: ', action) if debug else None

        self.removal_step += 1
        self.number_debris_left -= 1
        self.dt_left -= action[1]
        # Get the radii of the OTV and target
        r1 = env.debris_list[self.current_removing_debris].a 
        r2 = env.debris_list[action[0]].a 
        self.dv_left -= Cv(r1, r2)
        print('radii' , r1, r2)
        # Update current removing debris after computing CB
        self.current_removing_debris = action[0]
        self.binary_flags[self.current_removing_debris] = 1





    def to_list(self):
        return [self.removal_step , self.number_debris_left , self.dv_left , self.dt_left , self.current_removing_debris] + self.binary_flags
        


        