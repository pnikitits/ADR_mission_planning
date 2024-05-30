from astropy import units as u
from poliastro.bodies import Earth
from src.simulator.CustomOrbit import Orbit
from poliastro.plotting import OrbitPlotter3D
import src.simulator.CustomManeuvres as CustomManeuvres
import copy
import numpy as np
import scipy.io
import pandas as pd

class Debris:
    def __init__(self , poliastro_orbit , norad_id):
        self.poliastro_orbit = poliastro_orbit
        self.norad_id = norad_id



class Simulator:
    def __init__(self , starting_index=1 , n_debris=10):
        # Initialise the debris dictionary and assign the otv to an Orbit
        self.debris_list = self.init_random_debris(n=n_debris) 
        # self.debris_list = self.debris_from_dataset(n=n_debris) #le dataset contient 320 debris
        check_debris = False
        if check_debris:
            for idx, target_debris in enumerate(self.debris_list):
                print('debris number: ', idx )
                print('/n ', )
                target_debris = target_debris.poliastro_orbit
                print(target_debris.a, target_debris.ecc, target_debris.inc, target_debris.raan, target_debris.argp, target_debris.nu)
            print('Starting index: ', starting_index)
    

        self.otv_orbit = copy.copy(self.debris_list[starting_index].poliastro_orbit)
        

    def simulate_action(self , action):
        """
        Input:
            action : (next_debris_norad_id , dt_given)
        Output:
            DV_required , DT_required

        Can expand to testing multiple strategies
        Updates all objects in simulator
        When propagating, add at the end the (dt from action - dt required)
        """
        DV_required , DT_required = self.strategy_1(action)
        return DV_required , DT_required


    def strategy_1(self , action, render=False, step_sec=5):
        """
        Strategy 1 defined in transfer strategies slides
        1. Inc
        2. Raan
        3. Hohmann
        """


        # Set the target from the action
        target_debris = self.debris_list[action[0]].poliastro_orbit

        # ---- Inclination change
        inc_change = CustomManeuvres.simple_inc_change(self.otv_orbit, target_debris)

        # Get the transfer time of the hoh_phas
        transfer_time = inc_change.get_total_time()

        # Apply the maneuver to the otv
        self.otv_orbit, inc_frames = self.otv_orbit.apply_maneuver_custom(inc_change, copy.deepcopy(self.debris_list) if render else None, step_sec=step_sec, render=render)

        # Propagate all debris to the end of the transfer
        for i , debris in enumerate(self.debris_list):
            self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(transfer_time)
        

        # ---- Raan change
        target_debris = self.debris_list[action[0]].poliastro_orbit
        raan_change = CustomManeuvres.simple_raan_change(self.otv_orbit, target_debris)

        # Get the transfer time of the hoh_phas
        transfer_time = raan_change.get_total_time()

        # Apply the maneuver to the otv
        self.otv_orbit, raan_frames = self.otv_orbit.apply_maneuver_custom(raan_change, copy.deepcopy(self.debris_list) if render else None, step_sec=step_sec, render=render)

        # Propagate all debris to the end of the transfer
        for i , debris in enumerate(self.debris_list):
            self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(transfer_time)
        

        # ---- Hohmann
        target_debris = self.debris_list[action[0]].poliastro_orbit
        hoh_change = CustomManeuvres.hohmann_with_phasing(self.otv_orbit, target_debris)

        # Get the transfer time of the hoh_phas
        transfer_time = hoh_change.get_total_time()

        # Apply the maneuver to the otv
        self.otv_orbit, hoh_frames = self.otv_orbit.apply_maneuver_custom(hoh_change, copy.deepcopy(self.debris_list) if render else None, step_sec=step_sec, render=render)

        # Propagate all debris to the end of the transfer
        for i , debris in enumerate(self.debris_list):
            self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(transfer_time)      

        # Total resources used
        total_dv = hoh_change.get_total_cost() + raan_change.get_total_cost() + inc_change.get_total_cost()
        min_time = hoh_change.get_total_time() + raan_change.get_total_time() + inc_change.get_total_time()


        # Propagate with the extra time after the action
        extra_time = action[1] * u.day - min_time
        # print(extra_time)
        # if extra_time.value > 0:
        #     self.otv_orbit = self.otv_orbit.propagate(extra_time)
        #     for i , debris in enumerate(self.debris_list):
        #         self.debris_list[i].poliastro_orbit = debris.poliastro_orbit.propagate(extra_time)

        if render:
            # Concat the dataframes
            location_frames_df = pd.concat([inc_frames , raan_frames , hoh_frames] , axis=0)
            return location_frames_df
        else:
            return total_dv , min_time



    def init_random_debris(self , n):
        """
        Output:
            list (norad_id , Orbit)
        """
        np.random.seed(42) # !!!
        
        debris_list = []

        np.random.seed(42)

        for norad_id in range(n):
            min_a = 6371 + 200
            max_a = 6371 + 10000
            a = np.random.uniform(min_a, max_a) * u.km
            ecc = 0 * u.one
            inc = np.random.uniform(0, 45) * u.deg
            raan = np.random.uniform(0, 45) * u.deg
            argp = 0 * u.deg
            nu = np.random.uniform(-180, 180) * u.deg

            debris = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
            debris_list.append(Debris(poliastro_orbit=debris , norad_id=norad_id))

        return debris_list
    
    def debris_from_dataset(self, n):
        """
        Transform n first debris from the dataset in Debris object
        Output:
            list (norad_id , Orbit) 
        """
        debris_list = []
        dataset = scipy.io.loadmat('data/TLE_iridium.mat')['TLE_iridium']

        # Select only favourable debris
        i=0
        while len(debris_list) < n+1:
            norad_id = dataset[0][i]
            a = dataset[6][i] * u.km 
            # ecc = dataset[3][i] * u.one
            ecc = 0 * u.one
            inc = dataset[1][i] * u.deg
            raan = dataset[2][i] * u.deg
            # argp = dataset[4][i] * u.deg
            argp = 0 * u.deg
            nu = (dataset[5][i] - 180) * u.deg

            if dataset[2][i] < 20:
                debris = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
                debris_list.append(Debris(poliastro_orbit=debris , norad_id=norad_id))
            
            i +=1

        
        # for i in range(n):
        #     norad_id = dataset[0][i]
        #     a = dataset[6][i] * u.km 
        #     ecc = dataset[3][i] * u.one
        #     inc = dataset[1][i] * u.deg
        #     raan = dataset[2][i] * u.deg
        #     argp = dataset[4][i] * u.deg
        #     nu = dataset[5][i] * u.deg

        #     debris = Orbit.from_classical(Earth, a, ecc, inc, raan, argp, nu)
        #     debris_list.append(Debris(poliastro_orbit=debris , norad_id=norad_id))
        
        return debris_list

