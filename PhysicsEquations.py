import numpy as np
#from State import State
from Debris import Debris

"""
Physics Equations :

G   gravitational constant (mu = G*M)
M   mass of central object
r1  radius of orbit 1
r2  radius of orbit 2

dv  hohmann boost (dv1 + dv2)
th  hohmann boost duration
"""

G = 6.67e-7
M = 5.97e24



def hohmann_dv(r1 , r2 , G=G , M=M):
    mu = G*M

    dv1 = np.sqrt(mu/r1)*(np.sqrt(2*r2/(r1+r2))-1)
    dv2 = np.sqrt(mu/r2)*(np.sqrt(2*r1/(r1+r2))-1)

    dv = dv1 + dv2
    return dv


def hohmann_time(r1 , r2 , G=G , M=M):
    mu = G*M
    th = np.pi * np.sqrt( ((r1+r2)**3) / (8*mu) )
    return th


def phase_time(otv , target , G=G , M=M):
    """
    Time to wait until at right angle diff to start the hohmann transfer
    du is angle when to start

    Correction de phase_time, assuming r1 < r2
    """
    mu = G*M

    r1 = otv.a
    r2 = target.a

    init_ang_1 = otv.mean_anomaly
    init_ang_2 = target.mean_anomaly

    angle_diff = init_ang_2 - init_ang_1
    du = delta_u(r1 , r2)

    if angle_diff < du:
        angle_diff += 2*np.pi

    ang_vel_1 = np.sqrt( mu/(r1**3) )
    ang_vel_2 = np.sqrt( mu/(r2**3) )
    
    dt = (du - angle_diff) / (ang_vel_2 - ang_vel_1)
    return dt


    """# A COMPLETER (abs dans l'histoire)
    d_ang = otv.mean_anomaly - target.mean_anomaly - delta_u(otv.a , target.a)

    if d_ang < 0:
        d_ang = 2*np.pi - d_ang
    
    dt = d_ang / (target.angular_velocity - otv.angular_velocity)
    return dt
    # A COMPLETER"""


def delta_u(r1 , r2):
    return np.pi * (1 - np.sqrt( (r1+r2) / (2*r2) ))



def Cv(action , state , G=G , M=M):
    # Get current debris from state.current_removing_debris -> r1
    current_d = state.current_removing_debris
    r1 = current_d.a # object 'Debris', assuming circular orbit (r=a)

    # Get target debris from action[0] -> r2
    target_d = action[0]
    r2 = target_d.r

    # Compute dv(r1 , r2)
    dv = hohmann_dv(r1=r1 , r2=r2 , G=G , M=M)

    # Return dv
    return dv

