import numpy as np
#from State import State
# from Debris import Debris

"""
Physics Equations :

G   gravitational constant (mu = G*M)
M   mass of central object
r1  radius of orbit 1
r2  radius of orbit 2

dv  hohmann boost (dv1 + dv2)
th  hohmann boost duration
"""

G = 6.67e-11 # N * M^2 / KG^2
M = 5.97e24  # KG


def hohmann_dv(r1 , r2 , G=G , M=M):
    # Indepent of moving up or down
    mu = G*M

    dv1 = np.sqrt(mu/r1)*(np.sqrt(2*r2/(r1+r2))-1)
    dv2 = np.sqrt(mu/r2)*(np.sqrt(2*r1/(r1+r2))-1)

    dv = dv1 + dv2
    return dv


def hohmann_time(r1 , r2 , G=G , M=M):
    # Indepent of moving up or down
    mu = G*M
    th = np.pi * np.sqrt( ((r1+r2)**3) / (8*mu) ) / 86400 #Convert to days
    return th


def phase_time(otv , target , G=G , M=M):

    return 0
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
        angle_diff += 360

    ang_vel_1 = np.sqrt( mu/(r1**3) )
    ang_vel_2 = np.sqrt( mu/(r2**3) )

    if ang_vel_1 == ang_vel_2:
        dt = 0
    else:
        # Angles in degrees and angular velocities in degrees/day
        dt = (du - angle_diff) / (ang_vel_2 - ang_vel_1)
    
    if dt>0 and r2 > r1:
        print('radii')
        print(r1, r2)
        print('angles')
        print(otv.mean_anomaly, target.mean_anomaly)
        print('delta u')
        print(du)
        print('phase time')
        print(dt)

    return dt


    """# A COMPLETER (abs dans l'histoire)
    d_ang = otv.mean_anomaly - target.mean_anomaly - delta_u(otv.a , target.a)

    if d_ang < 0:
        d_ang = 2*np.pi - d_ang
    
    dt = d_ang / (target.angular_velocity - otv.angular_velocity)
    return dt
    # A COMPLETER"""


def delta_u(r1 , r2):
    # Return in degrees
    return np.rad2deg(np.pi * (1 - np.sqrt( (r1+r2) / (2*r2) )))



def Cv(r1, r2 , G=G , M=M):
    # Compute dv(r1 , r2)
    dv = hohmann_dv(r1=r1 , r2=r2 , G=G , M=M)

    # Return dv
    return dv

