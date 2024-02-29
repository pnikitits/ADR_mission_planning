from Debris import Debris
import numpy as np


def simple_phase(object:Debris , target_anomaly):
    """
    Calculates phasing time to reach a static point on the orbit

    All in deg/day
    """
    d_ma = target_anomaly - object.mean_anomaly

    if d_ma < 0:
        d_ma += 360
        
    phase_t =  d_ma / object.angular_velocity

    return phase_t # time in days


# def combined_inc_raan_dv(otv:Debris , target:Debris):
#     current_v = otv.angular_velocity * otv.a * 86400 # M / SEC

#     # convert to rad here
#     # TODO ADD MODULO HERE? Getting negative values out
#     initial_i = np.deg2rad(otv.inclination)
#     target_i = np.deg2rad(target.inclination)
#     initial_raan = np.deg2rad(otv.raan)
#     target_raan = np.deg2rad(target.raan)

#     d_i = target_i - initial_i
#     d_raan = target_raan - initial_raan
#     total_dv = current_v * (d_i + np.sin(initial_i) * d_raan)

#     return total_dv


def combined_inc_raan_dv(otv:Debris , target:Debris):

    v1 = otv.angular_velocity * otv.a / 86400**2 # M / SEC # CHECK THIS

    raan1 = np.deg2rad(otv.raan)
    raan2 = np.deg2rad(target.raan)
    inc1 = np.deg2rad(otv.inclination)
    inc2 = np.deg2rad(target.inclination)

    d_raan = raan2 - raan1
    theta = np.arccos( np.cos(d_raan)*np.sin(inc1)*np.sin(inc2) + np.cos(inc1)*np.cos(inc2) )
    return 2*v1*np.sin(theta/2)