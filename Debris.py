"""
https://github.com/JerrettYang/Iridium-33-17126UTC
"""
import numpy as np
from InPlaneEquations import G , M
from astropy import units as u


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

        # Init angular velocity
        self.angular_velocity = self.init_velocity()



    def update(self , dt):
        # Update position after timestep
        self.mean_anomaly += self.angular_velocity * dt
        self.mean_anomaly = self.mean_anomaly.to(u.deg)%(360*u.deg)

    def init_velocity(self):
        # Find the angular velocity
        a = self.a.to(u.m)
        rad_vel = np.sqrt(G*M / (a**3)) * u.rad
        return rad_vel
    

    def __repr__(self) -> str:
        return f"\033[36ma = {self.a.value} << u.km\necc = {self.eccentricity} << u.one\ninc = {self.inclination.value} << u.deg\nraan = {self.raan.value} << u.deg\nargp = {self.arg_perigee} << u.deg\nnu = {self.mean_anomaly.value - 180} << u.deg\033[0m"
        #return (f"Orbit(Earth , inc={self.inclination}, raan={self.raan}, eccentricity={self.eccentricity}, arg_perigee={self.arg_perigee}, mean_anomaly={self.mean_anomaly}, a={self.a})")