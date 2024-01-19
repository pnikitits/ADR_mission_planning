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

        # Angle between orbital plane and Earth equator plane - DEGREES
        self.inclination = inclination

        # Right ascension of ascending node (orientation of orbital plane) - DEGREES
        self.raan = raan

        # 0 for circular orbit, 0<e<1 for ellipic orbit - DEGREES
        self.eccentricity = eccentricity

        # Location of closest approach wrt ascending node - DEGREES
        self.arg_perigee = arg_perigee

        # Angular position in orbit (from perigee) - DEGREES
        self.mean_anomaly = mean_anomaly

        # Semi-major axis - converted to METRES
        self.a = a*1000

        # Radar cross section - M^2
        self.rcs = rcs

        # Init angular velocity - DEGREES / DAY
        self.angular_velocity = self.init_velocity()



    def update(self , dt):
        # Update position after timestep
        self.mean_anomaly += self.angular_velocity * dt
        self.mean_anomaly = self.mean_anomaly%360

    def init_velocity(self):
        # Find the angular velocity -  DEGREES / DAY
        rad_vel = np.sqrt(G*M / (self.a**3)) * 86400 # convert seconds to days
        return np.rad2deg(rad_vel)
    

    # def __repr__(self) -> str:
    #     return (f"Orbit(inclination={self.inclination}, raan={self.raan}, "
    #             f"eccentricity={self.eccentricity}, arg_perigee={self.arg_perigee}, "
    #             f"mean_anomaly={self.mean_anomaly}, a={self.a})")