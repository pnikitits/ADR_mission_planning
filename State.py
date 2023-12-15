import numpy as np
from PhysicsEquations import Cv

class State:
    def __init__(self , removal_step , total_n_debris , dv_max , dt_max , first_debris):
        self.removal_step = removal_step
        self.number_debris_left = total_n_debris
        self.dv_left = dv_max
        self.dt_left = dt_max
        self.current_removing_debris = first_debris
        self.binary_flags = np.zeros(total_n_debris).tolist

    def transition_function(self , action):
        self.removal_step += 1
        self.number_debris_left -= 1
        self.dv_left -= Cv(action , self)
        self.dt_left -= action[1]
        self.current_removing_debris = action[0]
        self.binary_flags[self.current_removing_debris] = 1
        