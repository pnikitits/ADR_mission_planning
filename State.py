import numpy as np
from Strat_1 import CV , strat_1_dv
from astropy import units as u

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
        
        #print(' --- Taking action: ', action) if debug else None

        self.removal_step += 1
        self.number_debris_left -= 1
        self.dt_left -= action[1]
        # Get the current and target debris
        otv = env.debris_list[self.current_removing_debris]
        target = env.debris_list[action[0]]

        dv_log , dt_log = strat_1_dv(otv , target , debug=True)
        print(f"--- Taking action {action}: 'dv={dv_log} , dt={dt_log}")

        print(f"{otv}")
        print(f"{target}")

        self.dv_left -= CV(otv, target).to(u.km/u.s).value
        # Update current removing debris after computing CB
        self.current_removing_debris = action[0]
        self.binary_flags[self.current_removing_debris] = 1





    def to_list(self):
        return [self.removal_step , self.number_debris_left , self.dv_left , self.dt_left , self.current_removing_debris] + self.binary_flags
        


        