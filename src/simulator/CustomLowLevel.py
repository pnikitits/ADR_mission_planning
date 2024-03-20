""" Low level maneuver implementations """

import numpy as np
from numba import njit as jit
from numpy import cross

#from poliastro._math.linalg import norm # _math ??
from poliastro.core.elements import coe_rotation_matrix, rv2coe, rv_pqw


@jit
def hohmann_any_angle(k, rv, r_f):
    r"""Calculate the Hohmann maneuver velocities and the duration of the maneuver.

    By defining the relationship between orbit radius:

    .. math::
        a_{trans} = \frac{r_{i} + r_{f}}{2}

    The Hohmann maneuver velocities can be expressed as:

    .. math::
        \begin{align}
            \Delta v_{a} &= \sqrt{\frac{2\mu}{r_{i}} - \frac{\mu}{a_{trans}}} - v_{i}\\
            \Delta v_{b} &= \sqrt{\frac{\mu}{r_{f}}} - \sqrt{\frac{2\mu}{r_{f}} - \frac{\mu}{a_{trans}}}
        \end{align}

    The time that takes to complete the maneuver can be computed as:

    .. math::
        \tau_{trans} = \pi \sqrt{\frac{(a_{trans})^{3}}{\mu}}

    Parameters
    ----------
    k : float
        Standard Gravitational parameter
    rv : numpy.ndarray, numpy.ndarray
        Position and velocity vectors
    r_f : float
        Final orbital radius

    """
    _, ecc, inc, raan, argp, nu = rv2coe(k, *rv)
    h_i = np.linalg.norm(cross(*rv)) # poliastro._math.linalg or np.linalg ?
    p_i = h_i**2 / k

    r_i, v_i = rv_pqw(k, p_i, ecc, nu)

    r_i = np.linalg.norm(r_i)
    v_i = np.linalg.norm(v_i)
    a_trans = (r_i + r_f) / 2

    dv_a = np.sqrt(2 * k / r_i - k / a_trans) - v_i
    dv_b = np.sqrt(k / r_f) - np.sqrt(2 * k / r_f - k / a_trans)

    dv_a = np.array([-np.sin(nu) * dv_a, np.cos(nu) * dv_a, 0])
    dv_b = np.array([np.sin(nu) * dv_b, -np.cos(nu) * dv_b, 0])

    rot_matrix = coe_rotation_matrix(inc, raan, argp)

    dv_a = rot_matrix @ dv_a
    dv_b = rot_matrix @ dv_b

    t_trans = np.pi * np.sqrt(a_trans**3 / k)

    return dv_a, dv_b, t_trans