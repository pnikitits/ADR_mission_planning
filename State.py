import numpy as np
from PhysicsEquations import Cv

class State:
    def __init__(self , removal_step , total_n_debris , dv_max , dt_max , first_debris):
        self.removal_step = removal_step
        self.number_debris_left = total_n_debris
        self.dv_left = dv_max
        self.dt_left = dt_max
        self.current_removing_debris = first_debris # Index
        self.binary_flags = np.zeros(total_n_debris).tolist()
        self.binary_flags[self.current_removing_debris] = 1

    def transition_function(self , env, action):
        self.removal_step += 1
        self.number_debris_left -= 1
        self.dt_left -= action[1]
        self.current_removing_debris = action[0]
        self.binary_flags[self.current_removing_debris] = 1

        # Get the radii of the OTV and target
        r1 = env.debris_list[self.current_removing_debris].a 
        r2 = env.debris_list[action[0]].a 

        self.dv_left -= Cv(r1, r2)


        