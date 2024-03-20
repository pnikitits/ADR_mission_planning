from astropy import units as u
import numpy as np

def delta_u(tof, target_orbit):
    """
        Calculate the angle at which the first burn should be applied
    """

    period = target_orbit.period
    mean_motion = 2*np.pi / period
    angle = (np.pi * u.rad) - mean_motion*(tof*u.s)*u.rad

    return angle << u.deg