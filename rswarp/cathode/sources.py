"""
Utilities for computing desired beam currents in Warp. This version uses separate functions.

Authors: Nathan Cook and Chris Hall
04/23/2017
"""

from __future__ import division
import numpy as np
from scipy.special import erf,erfinv
from rswarp.cathode import injectors


# Specify constants
from scipy.constants import e, m_e, c, k, epsilon_0
kb_eV = 8.6173324e-5  # Bolztmann constant in eV/K
kb_J = k  # Boltzmann constant in J/K
m = m_e  # mass of electron


##########################################
###### SUPPORT FUNCTIONS
##########################################

def cl_limit(cathode_phi, anode_wf, grid_bias, plate_spacing):
    """
    Compute the (cold) Child-Langmuir limit using grid geometry.

    Arguments:
        cathode_phi (float)      : cathode work function in V
        anode_wf (float)        : anode work function in V
        grid_bias (float)       : voltage on grid electrodes in V
        plate_spacing (float)   : distance between cathode and anode in m

    Returns:
        cl_current (float)      : Child-Langmuir current in A/m^2

    """

    # Compute vacuum level prior to computing current
    vacuum_level = cathode_phi - anode_wf + grid_bias
    cl_limit = 4. * epsilon_0 / 9. * np.sqrt(2. * e / m_e) * abs(vacuum_level)**(3./2.) / plate_spacing**2

    return cl_limit


def j_rd(T, phi):
    """Returns the Richardson-Dushmann thermionic emission given a temperature
    and effective work function. Constant coefficient of emission (A) is assumed.

    Arguments:
        T (float)   : temperature of the cathode in K
        phi (float) : work function of the cathode in eV

    Returns:
        J (float)   : current density in Amp/m^2

    """

    A = 1.20e6  # amp/m^2/degK^2

    return A*T**2*np.exp(-1.*phi/(kb_eV*T))
    
    
def j_sl(Te, Tr, phi):
    """Returns the Saha-Langmuir ion emission given a temperature
    and effective work function``. See Page 182 of Rasor (eq. 37).

    Arguments:
        Te (float)   : temperature of the cathode in K
        Tr (float)  : temperature of the reservoir in K
        phi (float) : neutral plasma workfunction in eV

    Returns:
        J (float)   : current density in Amp/m^2

    """
    Vi = 3.9 #Ionization potential of Cesium in eV
    h = 0.75 #eV - empirical - see page 180 of Rasor
    A = 1.20e6  # amp/m^2/degK^2
    D = 1e27 #cm^-2
    
    #rate of arrival of Cs atoms
    mu = D*np.exp(-1.*h/(kb_eV*Tr))
    
    return e*mu/(1.+2*np.exp((Vi-phi)/(kb_eV*Te)))



def get_MB_velocities(n_part, T):
    """Return a distribution of particle velocities representing a Maxwell-Boltzmann
    distribution.

    Each velocity distribution is a temperature dependent Gaussian. Due to the geometry
    of the cathode, negative longitudinal velocities are discarded. The individual
    velocity distributions are assumed to be independent, and hence there is no covariance
    between them.

    Arguments:
        n_part (int)        : number of particles to be used in the distribution
        T (float)           : temperature of the cathode in K

    Returns:
        pos_output (ndArray)    : n_part x 3 array with (vx,vy,vz) values for each particle

    """

    var_xy = kb_J*T/m  # Define the variance of the distribution in the x,y planes
    var_z = 2*kb_J*T/m  # Variance in z plane is twice that in the horizontal
    var_vs = np.asarray([var_xy,var_xy,var_z])
    mean = [0,0,0]  # Each distribution has a native mean of 0.
    cov = np.multiply(var_vs,np.identity(3))  # distributions are assumed to be independent
    # mean_vz = np.sqrt(2*var/np.pi)  # compute this from all-positive component of distribution

    # Additional values are computed so that tuples with negative vz can be discarded
    flag_array_full = False
    while not flag_array_full:
        output = np.random.multivariate_normal(mean,cov,int(np.round(n_part*3)))
        pos_output = output[np.where(output[:,2] > 0.)[0]]
        if pos_output.shape[0] >= n_part:
            flag_array_full = True

    return pos_output[:n_part]


def compute_expected_velocity(T):
    """
    Returns the expected value of the longitudinal velocity for the cathode with temperature T. Note that the
    expected value <v> is given by <v> = (2/pi)*sqrt(variance).
    """

    var_z = 2*kb_J*T/m

    return (2./np.pi)*np.sqrt(var_z)


def compute_cutoff_beta(T, frac=0.99):
    """Returns the velocity for which the fraction frac of a beam emitted from a thermionic
    cathode with temperature T move more slowly in the longitudinal (z) direction.

    Arguments:
        T (float)               : temperature of the cathode in K
        frac (Optional[float])  : Fraction of beam with vz < cutoff. Defaults to 0.99.

    Returns:
        beta_cutoff (float)     : cutoff velocity divided by c.

    """

    sigma = np.sqrt(2*kb_J*T/m)  # effective sigma for half-Gaussian distribution

    multiplier = erfinv(frac)  # the multiplier on sigma accounting for frac of the distribution

    return multiplier*sigma/c


def compute_crossing_fraction(cathode_temp, phi, zmesh):
    """
    Returns the % of particles expected to cross the gap

    Arguments:
        cathode_temp    (float) : cathode temperature in K
        phi             (scipy) : Interpolation (scipy.interpolate.interpolate.interp1d) of Phi from initial solve
        zmesh          (ndArray) : 1D array with positions of the z-coordinates of the grid

    Returns:
        e_frac          (float) : fraction of beam that overcomes the peak potential barrier phi
    """

    # Conditions at cathode edge
    var_xy = kb_J*cathode_temp/m  # Define the variance of the distribution in the x,y planes
    var_z = 2*kb_J*cathode_temp/m  # Variance in z plane is twice that in the horizontal

    #Phi is in eV
    vz_phi = np.sqrt(2.*e*np.max(phi(zmesh))/m_e)

    #cumulative distribution function for a Maxwellian
    CDF_Maxwell = lambda v, a: erf(v/(np.sqrt(2)*a)) - np.sqrt(2./np.pi)*v*np.exp(-1.*v**2/(2.*a**2))/a

    e_frac = CDF_Maxwell(vz_phi,np.sqrt(var_z)) #fraction with velocity below vz_phi

    return (1.-e_frac)*100. #Multiply by 100 to get % value


def compute_expected_time(beam, cathode_temp,Ez, zmin, zmax, dt):
    """
    Returns the expected time of flight for a particle with average initial velocity across the gap.

    Arguments:
        beam            (Warp)  : warp beam object (e.g. Species() call)
        cathode_temp    (float) : cathode temperature in K
        Ez              (scipy) : Interpolation (scipy.interpolate.interpolate.interp1d) of Ez field from initial solve
        zmin            (float) : Minimum z-value in domain
        zmax            (float) : Maximum z-value in domain
        dt              (float) : step size in time (specified by top.dt)

    Returns:
        t_curr (float)     : final time value at which simulated particle reaches right boundary
    """

    # Conditions at cathode edge
    z0 = zmin
    v0 = compute_expected_velocity(cathode_temp)
    t0 = 0

    # Conditions for initialization of algorithm
    z_curr = z0
    v_curr = v0 + ((beam.charge/beam.mass)*Ez(z0+v0*0.5*dt)*0.5*dt)  # offset velocity by a half step
    t_curr = t0

    while z_curr < zmax:
        # update velocity-this could be technically offset by 0.5v0dt
        v_curr = v_curr + (beam.charge/beam.mass)*Ez(z_curr)*dt
        # Update z
        z_curr = z_curr + v_curr*dt
        # Update t to get traversal time
        t_curr = t_curr + dt

        assert z_curr > 0, \
            'Expected crossing time could not be calculated. Particle with expected velocity {:.3e} fails to cross specified geometry'.format(v0)

    return t_curr

##########################################
###### INSTANTIATIONS
##########################################


def constant_current(beam, channel_width, z_part_min, ptcl_per_step):
    """
    Instantiate a beam with constant, user-specified current and zero temperature.

    Arguments:
        current (float)         : beam current in Amperes
        a0      (float)         : X-plane source radius in m
        b0      (float)         : Y-plane source radius in m FOR 3D SIMULATIONS
        channel_width   (float) : width of domain in x/y plane
        z_part_min      (float) : z coordinate of particles injected - provides separation from left boundary
        ptcl_per_step   (int)   : number of macro particles injected per step

    """
    # top.inject = 1 must be specified in main script

    # fixed cathode temperature
    myInjector = injectors.injectorUserDefined(beam, 4.0, channel_width, z_part_min, ptcl_per_step)

    installuserinjection(myInjector.inject_electrons)

    # These must be set for user injection
    top.ainject = 1.0
    top.binject = 1.0


# def child_langmuir_current(current,cathode_phi,anode_wf,grid_bias):
# def child_langmuir_current(current, a0, b0):
def child_langmuir_current():
    """
    Instantiate a beam with (cold) Child-Langmuir limited current. Current must be computed using
    available geometry.
    """

    # top.inject = 2 must be specified in main script
    # beam.ibeam  = current
    # beam.a0     = a0
    # beam.b0     = b0
    # cold beam approximation
    # beam.vthz   = 0
    # beam.vthperp= 0
    w3d.l_inj_exact = True  # this is needed for top.inject=2


def thermionic_current(beam, cathode_temp, channel_width, z_part_min, ptcl_per_step):
    """Instantiate a beam with the Richardson-Dushmann current and a thermal distribution of velocities.

    Arguments:
        beam            (Warp)  : warp beam object (e.g. Species() call)
        current         (float) : beam current in A
        cathode_temp    (float) : cathode temperature in K
        cathode_phi     (float) : cathode work function in V
        cathode_area    (float) : cathode surface area in m^2
        a0              (float) : X-plane source radius in m
        b0              (float) : Y-plane source radius in m FOR 3D SIMULATIONS
        channel_width   (float) : width of domain in x/y plane
        z_part_min      (float) : z coordinate of particles injected - provides separation from left boundary
        ptcl_per_step   (int)   : number of macro particles injected per step


    """

    # top.inject = 6 must be specified in main script

    myInjector = injectors.injectorUserDefined(beam, cathode_temp, channel_width, z_part_min, ptcl_per_step)

    installuserinjection(myInjector.inject_thermionic)

    # These must be set for user injection
    top.ainject = 1.0
    top.binject = 1.0
