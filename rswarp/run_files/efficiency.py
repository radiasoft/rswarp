from scipy.constants import e, m_e, k, h, physical_constants
import numpy as np
k_ev = physical_constants['Boltzmann constant in eV/K'][0]
sigma_sb = physical_constants['Stefan-Boltzmann constant'][0] * 1e-4
L = 2.44e-8  # Lorentz number in units of W*Ohms/K**2

# TODO: Need to account for particles travelling back to the emitter
# Current in Amps
# Current density in A/cm**2
# Temperature in Kelvin
# Work function in eV
# Resistance in Ohms

# alpha and rho are an average for common materials


tec_parameters = {
    'A_em': [False, 'Emitter area (cm**2)'],
    'R_ew': [False, 'Emitter wiring resistance (Ohms)'],
    'P_em': [False, 'Emitter electron power (W/cm**2)'],
    'J_em': [False, 'Emitter current density (A/cm**2)'],
    'phi_em': [False, 'Emitter work function (eV)'],
    'T_em': [False, 'Emitter temperature (K)'],
    'R_cw': [False, 'Collector wiring resistance (Ohms)'],
    'J_coll': [False, 'Collector current density (A/cm**2)'],  # Will set analytically based on collector temp
    'phi_coll': [False, 'Collector work function (eV)'],
    'T_coll': [False, 'Collector temperature (K)'],
    'J_ec': [False, 'Current from emitter that reaches collector (A/cm**2)'],
    'x_struts': [False, 'Number of struts that intersect the x-axis'],
    'y_struts': [False, 'Number of struts that intersect the y-axis'],
    'V_grid': [False, 'Bias on grid relative to the emitter (V)'],
    'J_grid': [False, 'Grid current density (A/cm**2)'],
    'grid_height': [False, 'Position of the grid relative to emitter, normalized by collector position'],
    'strut_width': [False, 'Size of the strut parallel to intersecting axis (m)'],
    'strut_height': [False, 'Size of the strut along the z-axis'],
    'emiss_eff': [0.1, 'Emissivity ()'],
    'T_env': [300., 'Ambient temperature (K)'],
    'L_wire': [1.0, 'Wire length for emitter/collector (m)'],
    'rho': [4.792857143e-8, 'Resistivity (Ohms/m)'],
    'alpha': [0.0044, 'Temperature coefficient of resistance (1/K)'],
    'occlusion': [False, 'Fractional occlusion of collector by grid'],
    'run_time': [False, 'Simulation run time (s)']
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

    A = 4 * np.pi * m_e * k ** 2 * e / h ** 3

    return A * T ** 2 * np.exp(-phi / (k_ev * T)) * 0.01**2


def calculate_resistance(T, L_wire, rho, alpha, T_ref=300., **kwargs):
    """
    Temperature dependent resistance model.
    Area will be the area of the collector/emitter. Provides reasonable scaling since the wire gauge will vary
    depending on current which in turn is a function of unit area of the system.
    Args:
        T: Temperature of wire (K)
        A: Area (cm**2) [Fixed at 1 cm**2 for now]
        L_wire: Length of wire (m)
        rho: Resistivity (Ohms/m)
        alpha: Reference resistance coefficient
        **kwargs: Catch unused arguments

    Returns:
        Resistance (Ohms)
    """

    L_wire, rho, alpha, A = L_wire[0], rho[0], alpha[0], 1.

    delta_T = T - T_ref

    R0 = rho * L_wire / (A / 10000. / 2.)
    R1 = R0 * (1 + alpha * delta_T)

    return R1


def calculate_occlusion(x_struts, y_struts, strut_width, A_em, **kwargs):
    """
    Calculate the fractional area of the collector occluded by the grid.
    Assumes square domain.
    Args:
        x_struts: Struts along x
        y_struts: Struts along y
        strut_width: Strut width (m)
        A_em: Emitter area (cm**2)

    Returns:
        Fractional occlusion
    """
    nx = x_struts[0]
    ny = y_struts[0]
    w = strut_width[0] * 1e2
    L = np.sqrt(A_em[0])

    fractional_occlusion = ((nx + ny) * L * w - (nx * ny) * w ** 2) / L ** 2

    return fractional_occlusion


def calculate_power_flux(velocity, weight, phi, run_time, A_em, **kwargs):
    """
    Calculate total power from array of particle velocities.
    Will return 'real' power computed based on macroparticle weight.
    Args:
        velocity: Array of particle velocities in m/s [N, 3]
        weight: Macroparticle weight
        phi: Work function of the emitter in eV
        run_time: Time over which velocities were collected
        A_em: Emitter area in cm**2
        **kwargs: Catch unused arguments

    Returns:
        Power in W/cm**2
    """
    run_time, A_em = run_time[0], A_em[0]
    v_sqr = velocity[:, 0]**2 + velocity[:, 1]**2 + velocity[:, 2]**2
    ke = 0.5 * m_e * np.sum(v_sqr)
    N = v_sqr.size
    E_tot = ke + phi * e * N
    print "Etot: {}".format(E_tot)
    return E_tot * weight / run_time / A_em


def calculate_efficiency(A_em, R_ew, J_em, P_em, phi_em, T_em,
               R_cw, J_ec, phi_coll, T_coll,
               emiss_eff, T_env, J_grid, V_grid,
               occlusion, run_time, **kwargs):
    """
    Calculate the TEC efficieny.
    All power terms should be calculated to give W/cm**2
    Based on S. Meir et al. J. Renewable Sustainable Energy 2013.
    Args:
        A_em: Emitter area (cm**2)
        R_ew: Emitter wiring resistance (Ohms)
        J_em: Emitter current density (A/cm**2)
        phi_em: Emitter work function (eV)
        T_em: Emitter temperature (K)
        R_cw: Collector wiring resistance (Ohms)
        phi_coll: Collector work function (eV)
        T_coll: Collector temperature (K)
        emiss_eff: Emissivity ()
        T_env: Ambient temperature (K)
        occlusion: Fractional occlusion of collector by grid ()
        **kwargs: Catch unused arguments

    Returns:
        Efficiency ()
    """

    A_em, R_ew, J_em, P_em, phi_em, T_em, \
    R_cw, J_ec, phi_coll, T_coll, \
    emiss_eff, T_env, J_grid, V_grid, occlusion, run_time = A_em[0], R_ew[0], J_em[0], P_em[0], phi_em[0], T_em[0], \
               R_cw[0], J_ec[0], phi_coll[0], T_coll[0], \
               emiss_eff[0], T_env[0], J_grid[0], V_grid[0], occlusion[0], run_time[0]

    t = 1. - occlusion
    J_coll = rd_current(phi_coll, T_coll)
    # Modify J_ec (emitter to collector current) for analytical collector produced current
    J_ec = J_ec - J_coll
    # P_ew
    P_ew = 0.5 * (L / (A_em * R_ew) * (T_em - T_env) ** 2 - (J_em - t * J_coll) ** 2 * A_em * R_ew)

    # P_r
    P_r = emiss_eff * sigma_sb * (T_em ** 4 - t * T_coll ** 4)

    # P_ec (electron cooling power)
    P_ec = P_em - t * J_coll * (phi_em + 2 * k_ev * T_coll)

    # P_load
    V_lead = J_ec * R_cw + (J_ec - t * J_coll) * R_ew
    V_load = (phi_em - phi_coll) - V_lead
    P_load = J_ec * V_load

    # P_gate
    P_gate = (J_grid * t + J_coll) * V_grid

    eta = (P_load - P_gate) / (P_ec + P_r + P_ew)

    return eta