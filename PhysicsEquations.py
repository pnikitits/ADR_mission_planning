import numpy as np
from State import State
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


def phase_time():
    pass


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

