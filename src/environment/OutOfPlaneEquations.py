from src.environment.Debris import Debris
import numpy as np
from astropy import units as u


def simple_phase(object:Debris , target_anomaly):
    """
    Calculates phasing time to reach a static point on the orbit

    All in deg/day
    """
    d_ma = target_anomaly.to(u.rad) - (object.mean_anomaly + object.arg_perigee).to(u.rad)

    if d_ma < 0:
        d_ma += 2*np.pi * u.rad
        
    phase_t =  d_ma / (object.angular_velocity).to(u.rad / u.s)

    return phase_t # [s]



def combined_inc_raan_dv(otv:Debris , target:Debris):

    v1 = (otv.angular_velocity * otv.a / u.rad).to(u.m / u.s)

    raan1 = otv.raan.to(u.rad)
    raan2 = target.raan.to(u.rad)
    inc1 = otv.inclination.to(u.rad)
    inc2 = target.inclination.to(u.rad)

    d_raan = raan2 - raan1
    theta = np.arccos( np.cos(d_raan)*np.sin(inc1)*np.sin(inc2) + np.cos(inc1)*np.cos(inc2) )
    return 2*v1*np.sin(theta/2) # [m/s]