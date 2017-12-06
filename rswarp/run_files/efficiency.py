from scipy.constants import e, m_e, k, h, physical_constants
from numpy import pi, exp
k_ev = physical_constants['Boltzmann constant in eV/K'][0]
sigma_sb = physical_constants['Stefan-Boltzmann constant'][0]
L = 2.44e-8  # Lorentz number in units of W*Ohms/K**2

# Current in Amps
# Current density in A/cm**2
# Temperature in Kelvin
# Work function in eV
# Resistance in Ohms

# alpha and rho are an average for common materials


tec_parameters_template = {
    'A_em': [0, 'Emitter area (cm**2)'],
    'R_ew': [0, 'Emitter wiring resistance (Ohms)'],
    'R_em': [0, 'Emitter resistance (Ohms)'],
    'J_em': [0, 'Emitter current density (A/cm**2)'],
    'phi_em': [0, 'Emitter work function (eV)'],
    'T_em': [0, 'Emitter temperature (K)'],
    'R_cw':  [0, 'Collector wiring resistance (Ohms)'],
    'J_coll': [0, 'Collector current density (A/cm**2)'],  # Will set analytically based on collector temp
    'phi_coll': [0, 'Collector work function (eV)'],
    'T_coll': [0, 'Collector temperature (K)'],
    'emiss_eff': [0, 'Emissivity ()'],
    'T_env': [0, 'Ambient temperature (K)'],
    'L_wire': [1.0, 'Wire length for emitter/collector (m)'],
    'rho': [4.792857143e-8, 'Resistivity (Ohms/m)'],
    'alpha': [0.0044, 'Temperature coefficient of resistance (1/K)'],
    'occlusion': [0, 'Fractional occlusion of collector by grid']
}


def rd_current(phi, T):
    """
    Thermionic emission current density based on Richardson-Dushman
    Args:
        phi: Work function (eV)
        T: Temperature (K)

    Returns:
        Current density in J/cm**2
    """

    A = 4 * pi * m_e * k ** 2 * e / h ** 3

    return A * T ** 2 * exp(-phi / (k_ev * T)) * 0.01**2


def calculate_resistance(T, A, L_wire, rho, alpha, **kwargs):
    """
    Temperature dependent resistance model.
    Area will be the area of the collector/emitter. Provides reasonable scaling since the wire gauge will vary
    depending on current which in turn is a function of unit area of the system.
    Args:
        T: Temperature of wire (K)
        A: Area (cm**2)
        L_wire: Length of wire (m)
        rho: Resistivity (Ohms/m)
        alpha: Reference resistance coefficient
        **kwargs: Catch unused arguments

    Returns:
        Resistance (Ohms)
    """

    L_wire, rho, alpha = L_wire[0], rho[0], alpha[0]

    T_ref = 300.0  # Reference temperature for rho
    delta_T = T - T_ref

    R0 = rho * L_wire / (A / 10000. / 2.)
    R1 = R0 * (1 + alpha * delta_T)

    return R1


def calculate_occlusion(nx, ny, w, L):
    """
    Calculate the fractional area of the collector occluded by the grid.
    Args:
        nx: Struts along x
        ny: Struts along y
        w: Strut width
        L: Strut length

    Returns:
        Fractional occlusion
    """
    fractional_occlusion = ((nx + ny) * L * w - (nx * ny) * w ** 2) / L ** 2

    return fractional_occlusion


def efficiency(A_em, R_ew, R_em, J_em, phi_em, T_em,
               R_cw, J_coll, phi_coll, T_coll,
               emiss_eff, T_env, occlusion, **kwargs):
    """
    Calculate the TEC efficieny.
    Based on S. Meir et al. J. Renewable Sustainable Energy 2013.
    Args:
        A_em: Emitter area (cm**2)
        R_ew: Emitter wiring resistance (Ohms)
        R_em: Emitter resistance (Ohms)
        J_em: Emitter current density (A/cm**2)
        phi_em: Emitter work function (eV)
        T_em: Emitter temperature (K)
        R_cw: Collector wiring resistance (Ohms)
        J_coll: Collector current density (A/cm**2)
        phi_coll: Collector work function (eV)
        T_coll: Collector temperature (K)
        emiss_eff: Emissivity ()
        T_env: Ambient temperature (K)
        occlusion: Fractional occlusion of collector by grid ()
        **kwargs: Catch unused arguments

    Returns:
        Efficiency ()
    """

    A_em, R_ew, R_em, J_em, phi_em, T_em, \
    R_cw, J_coll, phi_coll, T_coll, \
    emiss_eff, T_env, occlusion = A_em[0], R_ew[0], R_em[0], J_em[0], phi_em[0], T_em[0], \
               R_cw[0], J_coll[0], phi_coll[0], T_coll[0], \
               emiss_eff[0], T_env[0], occlusion[0]


    t = 1. - occlusion
    J_coll = rd_current(phi_coll, T_em)  # TODO: Decide where to set variables with dependencies like J_coll
    J_load = J_em - J_coll

    # P_ew
    P_ew = 0.5 * (L / (A_em * R_em) * (T_em - T_env) ** 2 - J_load ** 2 * A_em * R_ew)

    # P_r
    P_r = emiss_eff * sigma_sb * (T_em ** 4 - t * T_coll ** 4)

    # P_ec
    P_ec = J_em / e * (phi_em + 2 * k_ev * T_em) - J_coll / e * (phi_em + 2 * k_ev * T_coll)

    # P_load
    V_load = (phi_em - phi_coll) / e - J_load * A_em * (R_ew + R_cw)
    P_load = J_load * V_load

    eta = P_load / (P_ec + P_r + P_ew)

    return eta