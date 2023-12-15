"""
https://github.com/JerrettYang/Iridium-33-17126UTC
"""
import numpy as np
from PhysicsEquations import G , M


class Debris:
    def __init__(self , norad , inclination , raan , eccentricity ,
                 arg_perigee , mean_anomaly , a , rcs):
        
        # ID
        self.norad = norad

        # Angle between orbital plane and Earth equator plane
        self.inclination = inclination

        # Right ascension of ascending node (orientation of orbital plane)
        self.raan = raan

        # 0 for circular orbit, 0<e<1 for ellipic orbit
        self.eccentricity = eccentricity

        # Location of closest approach wrt ascending node
        self.arg_perigee = arg_perigee

        # Angular position in orbit (from perigee)
        self.mean_anomaly = mean_anomaly

        # Semi-major axis
        self.a = a

        # Radar cross section
        self.rcs = rcs

        self.angular_velocity = self.init_velocity()



    def update(self , dt):
        self.mean_anomaly += self.angular_velocity * dt
        

    def init_velocity(self):
        # Find the angular velocity
        return np.sqrt(G*M / (self.a**3))
    

    def __repr__(self) -> str:
        return f"Debris [{self.norad} , {self.inclination} , {self.raan} , {self.eccentricity} , 
                         {self.arg_perigee} , {self.mean_anomaly} , {self.a} , {self.rcs}]"
        
