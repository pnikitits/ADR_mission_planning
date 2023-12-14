import numpy as np
"""
Physics Equations

G   gravitational constant (mu = G*M)
M   mass of central object
r1  radius of orbit 1
r2  radius of orbit 2

dv  hohmann boost (dv1 + dv2)
th  hohmann boost duration

"""

def hohmann_dv(G,M,r1,r2):
    mu = G*M

    dv1 = np.sqrt(mu/r1)*(np.sqrt(2*r2/(r1+r2))-1)
    dv2 = np.sqrt(mu/r2)*(np.sqrt(2*r1/(r1+r2))-1)

    dv = dv1 + dv2
    return dv


def hohmann_time(G,M,r1,r2):
    mu = G*M
    th = np.pi * np.sqrt( ((r1+r2)**3) / (8*mu) )
    return th


def phase_time():
    pass


def Cv():
    return

