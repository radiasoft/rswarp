from scipy.constants import e, m_e, c, epsilon_0
from scipy.constants import Boltzmann as k_b
import numpy as np


def electric_field(r, Q):
    """
    Electric field at distance `r` away from point particle of charge `Q`
    
    Returns absolute value in V/m
    """
    K = 1. / (4 * np.pi * epsilon_0)
    return K * Q * e / r

def normalized_velocity_kick(r0, dt, v_rms, Q):
    """
    For softening parameter `r0` and a discrete time step `dt`
    calculate the ratio of the velocity change induced on an electron by
    an ion of charge `Q` to the `v_rms` the rms velocity spread of a group 
    of electrons.
    """
    dvn = electric_field(r0, Q)
    dvn *= dt / v_rms
    dvn *= e / m_e
    
    return dvn

def debye_length(T, n):
    """
    For `T` in K and `n` in m^-3:
    Calculate Debye length of an electron plasma
    as sqrt(eps_0 * k_b * T / (e^2 * n))
    """
    debye_length = np.sqrt(epsilon_0 * k_b * T / \
                           (e**2 * n))
    
    return debye_length

def plasma_frequency(n):
    """
    For `n` in m^-3 calculate the plasma frequency for an electron plasma
    sqrt(n * e^2 / (m_e * eps_0))
    """
    plasma_freq = np.sqrt(n * e**2 / (m_e * epsilon_0))
    
    return plasma_freq
    