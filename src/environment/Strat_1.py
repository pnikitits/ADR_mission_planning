from src.environment.Debris import Debris
import numpy as np
from src.environment.InPlaneEquations import G , M , phase_time
from src.environment.OutOfPlaneEquations import simple_phase, combined_inc_raan_dv
from src.environment.InPlaneEquations import hohmann_dv, hohmann_time
from astropy import units as u

mu = G*M

"""
Strategy 1

1. Inclination change
2. Raan change
3. Hohmann

All dv in M / S
"""

def strat_1_dv(otv:Debris , target:Debris, debug = False):
    """
    Main function for strat_1, returns the total dv and dt used
    """
    
    # Combined dv
    dv_inc_raan = combined_inc_raan_dv(otv=otv , target=target)

    # dv steps
    # 1. Inclination change
    dt_i = phase_for_i(otv=otv)
    otv.update(dt=dt_i) # ma 0 or 180 to check
    target.update(dt=dt_i) # propagate target position
    
    # 2. Raan change
    dt_raan = phase_for_raan(otv=otv)
    otv.update(dt=dt_raan) # ma 90 or 270 to check
    target.update(dt=dt_raan) # propagate target position

    # 3. Hohmann
    dv_hohmann = hohmann_dv(otv.a , target.a) # in M / SEC
    phase_hohmann = phase_time(otv=otv , target=target)
    dt_hohmann = hohmann_time(otv.a , target.a)

    # Total
    dv_total = dv_inc_raan + dv_hohmann
    print(f"dv_inc_raan={dv_inc_raan} , dv_hohmann={dv_hohmann}") if debug else None
    print(f"dt_i {dt_i}") if debug else None
    print(f"dt_raan {dt_raan}") if debug else None
    print(f"dt hohmann {phase_hohmann + dt_hohmann}") if debug else None

    dt_total = dt_i + dt_raan + phase_hohmann + dt_hohmann

    return dv_total , dt_total

def CV(otv:Debris , target:Debris):
    cv, _ = strat_1_dv(otv , target)
    return cv

def DT_required(otv:Debris , target:Debris):
    _, dt = strat_1_dv(otv , target)
    return dt



def phase_for_i(otv:Debris):
    t_0 = simple_phase(object=otv , target_anomaly=0*u.deg)
    t_180 = simple_phase(object=otv , target_anomaly=180*u.deg)
    return min(t_0 , t_180)


def phase_for_raan(otv:Debris):
    t_90 = simple_phase(object=otv , target_anomaly=90*u.deg)
    t_270 = simple_phase(object=otv , target_anomaly=270*u.deg)
    return min(t_90 , t_270)