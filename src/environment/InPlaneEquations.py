import numpy as np
from astropy import units as u
from astropy import constants as const

"""
Physics Equations :

G   gravitational constant (mu = G*M)
M   mass of central object
r1  radius of orbit 1
r2  radius of orbit 2

dv  hohmann boost (dv1 + dv2)
th  hohmann boost duration
"""

G = const.G
M = const.M_earth


def hohmann_dv(r1 , r2):
    # Indepent of moving up or down
    mu = G*M

    dv1 = np.sqrt(mu/r1)*(np.sqrt(2*r2/(r1+r2))-1)
    dv2 = np.sqrt(mu/r2)*(np.sqrt(2*r1/(r1+r2))-1)

    dv = dv1 + dv2
    return dv.to(u.m/u.s)


def hohmann_time(r1 , r2):
    # Indepent of moving up or down
    mu = G*M
    th = np.pi * np.sqrt( ((r1+r2)**3) / (8*mu) )
    return th.to(u.s)




def phase_time(otv , target):
    
    r1 = otv.a.to(u.m)
    r2 = target.a.to(u.m)
    ang_1 = otv.mean_anomaly.to(u.rad)
    ang_2 = target.mean_anomaly.to(u.rad)
    
    # Check orbits
    if r1 == r2:
        return 0
    
    # delta_u
    du = delta_u(r1 , r2)
    if r1 > r2:
        reverse_du = delta_u(r2 , r1)
        du = reverse_du

    # angle diff
    angle_diff = ang_2 - ang_1
    
    if r1 < r2 and angle_diff < du:
        angle_diff += 2*np.pi * u.rad
    elif r1 > r2 and angle_diff > du:
        angle_diff -= 2*np.pi * u.rad
    
    # angle velocity
    ang_vel_1 = otv.angular_velocity.to(u.rad / u.s)
    ang_vel_2 = target.angular_velocity.to(u.rad / u.s)

    # phasing time
    dt = (du - angle_diff) / (ang_vel_2 - ang_vel_1)
    return dt






def delta_u(r1 , r2):
    return np.pi * (1 - np.sqrt( (r1+r2) / (2*r2) )) * u.rad



def Cv(r1, r2):
    # Compute dv(r1 , r2)
    dv = hohmann_dv(r1=r1 , r2=r2)

    # Return dv
    return dv