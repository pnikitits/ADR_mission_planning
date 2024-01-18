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





def phase_time(otv , target , G=G , M=M , debug_msg=False):
    
    mu = G*M                    # N * M^2 / KG
    r1 = round(otv.a , 2)                  # M
    r2 = round(target.a , 2)               # M
    ang_1 = round(otv.mean_anomaly , 2)    # DEG
    ang_2 = round(target.mean_anomaly , 2) # DEG

    #if r1 > r2:
    #    debug_msg = True


    print("\n--- Phase Time ---\n") if debug_msg else None
    # Check orbits
    if r1 == r2:
        print("SAME ORBIT") if debug_msg else None
        print("\n------ END -------\n") if debug_msg else None
        return 0
    elif r1 < r2:
        print("GO UP:" , r1 , r2) if debug_msg else None
    elif r1 > r2:
        print("GO DOWN:" , r1 , r2) if debug_msg else None
    

    # delta_u
    du = delta_u(r1 , r2)
    print("DU:" , round(du , 2)) if debug_msg else None
    if r1 > r2:
        reverse_du = delta_u(r2 , r1)
        print("Reverse DU:" , round(reverse_du , 2)) if debug_msg else None
        du = reverse_du


    # angle diff
    angle_diff = round(ang_2 - ang_1 , 2)
    
    if angle_diff > 0:
        print(f"ang_1 < ang_2 : {ang_1}° < {ang_2}° | diff : {angle_diff}") if debug_msg else None
    elif angle_diff < 0:
        print(f"ang_1 > ang_2 : {ang_1}° > {ang_2}° | diff : {angle_diff}") if debug_msg else None
    elif angle_diff == 0:
        print(f"same angle | diff : {angle_diff}") if debug_msg else None

    if r1 < r2 and angle_diff < du:
        angle_diff += 360
        print("modified angle diff +360° :" , angle_diff) if debug_msg else None
    elif r1 > r2 and angle_diff > du:
        angle_diff -= 360
        print("modified angle diff -360° :" , angle_diff) if debug_msg else None
    





    # angle velocity
    ang_vel_1 = round(otv.angular_velocity , 2)    # Deg / Day
    ang_vel_2 = round(target.angular_velocity , 2) # Deg / Day
    print(f"ang_vel: {ang_vel_1} | {ang_vel_2}") if debug_msg else None

    # phasing time
    dt = (du - angle_diff) / (ang_vel_2 - ang_vel_1)
    if dt < 0:
        print(f"--------------------------------------------------------> phasing time: {round(dt , 2)}") if debug_msg else None
    else:
        print(f"phasing time: {round(dt , 2)}") if debug_msg else None

    print("\n------ END -------\n") if debug_msg else None
    return dt






def delta_u(r1 , r2):
    # Return in degrees
    return np.rad2deg(np.pi * (1 - np.sqrt( (r1+r2) / (2*r2) )))



def Cv(r1, r2 , G=G , M=M):
    # Compute dv(r1 , r2)
    dv = hohmann_dv(r1=r1 , r2=r2 , G=G , M=M)

    # Return dv
    return dv



"""
1. Boost -> drift orbit
    (Hohmann_1 + inclination_1) dv_comb_1

2. Boost pour rester sur drift orbit
    (Hohmann_1_p) dv_H1_p

3. Drift

4. Boost -> target orbit
    (Hohmann_2 + inclination_2) dv_comb_2

5. Boost pour rester sur target orbit
    (Hohmann_2_p) dv_H2_p
"""


def dv_H (r1 , r2 , mu):
    """
    Hohmann 1st boost
    """
    return np.sqrt(mu/r1) * ( np.sqrt( 2*r2 / (r1+r2) ) - 1 )


def dv_H_p (r1 , r2 , mu):
    """
    Hohmann 2nd boost
    """
    return np.sqrt(mu/r2) * ( 1 - np.sqrt( 2*r1 / (r1+r2) ) )


def dv_comb (r1 , r2 , mu , v1 , dI):
    """
    Hohmann boost with inclination change
    """
    v2 = v1 + dv_H(r1=r1 , r2=r2 , mu=mu)
    return np.sqrt( v1**2 + v2**2 - 2*v1*v2 * np.cos(dI) )
    


def dv_sum (debris_1 , debris_2 , ad , Id):
    mu = G*M

    r_initial = debris_1.a
    I_initial = debris_1.inclination

    r_final = debris_2.a
    I_final = debris_2.inclination


    # 1.
    dv_comb_1 = dv_comb()

    # 2.
    dv_H1_p = dv_H_p()

    # 4.
    dv_comb_2 = dv_comb()

    # 5.
    dv_H2_p = dv_H_p()


    total = dv_comb_1 + dv_H1_p + dv_comb_2 + dv_H2_p
    pass