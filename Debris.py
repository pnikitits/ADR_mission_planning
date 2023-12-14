"""
https://github.com/JerrettYang/Iridium-33-17126UTC
"""

class Debris:
    def __init__(self , norad , inclination , raan , eccentricity ,
                 arg_perigee , mean_anomaly , a , rcs):
        self.norad = norad
        self.inclination = inclination
        self.raan = raan
        self.eccentricity = eccentricity
        self.arg_perigee = arg_perigee
        self.mean_anomaly = mean_anomaly
        self.a = a
        self.rcs = rcs
        
