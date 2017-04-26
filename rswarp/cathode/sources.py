"""
Utilities for computing desired beam currents in Warp. This version uses separate functions.

Authors: Nathan Cook and Chris Hall
04/23/2017
"""

from __future__ import division
import numpy as np
import scipy
import injectors


#Specify constants
from scipy.constants import e, m_e, c, k
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e #mass of electron


##########################################
###### SUPPORT FUNCTIONS
##########################################    

def cl_limit(cathode_phi, anode_wf, grid_bias, plate_spacing):
    '''
    Compute the (cold) Child-Langmuir limit using grid geometry.
    
    Arguments:
        cathode_wf (float)      : cathode work function in V
        anode_wf (float)        : anode work function in V
        grid_bias (float)       : voltage on grid electrodes in V
        plate_spacing (float)   : distance between cathode and anode in m
    
    Returns:
        cl_current (float)      : Child-Langmuir current in A
    
    '''    
    
    #Compute vacuum level prior to computing current
    vacuum_level = cathode_wf - anode_wf + grid_bias
    cl_limit = 4. * eps0 / 9. * np.sqrt(2. * echarge / emass) * abs(vacuum_level)**(3./2.) / plate_spacing**2
    
    return cl_limit
    
    

def j_rd(T, phi):
    '''Returns the Richardson-Dushmann thermionic emission given a temperature 
    and effective work function. Constant coefficient of emission (A) is assumed.

    Arguments:
        T (float)   : temperature of the cathode in K
        phi (float) : work function of the cathode in eV

    Returns:
        J (float)   : current density in Amp/m^2
    
    '''

    A = 1.20e6 #amp/m^2/degK^2

    return A*T**2*np.exp(-1.*phi/(kb_eV*T))

def get_MB_velocities(n_part, T):
    '''Return a distribution of particle velocities representing a Maxwell-Boltzmann 
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

    '''

    var_xy = kb_J*T/m #Define the variance of the distribution in the x,y planes
    var_z = 2*kb_J*T/m #Variance in z plane is twice that in the horizontal
    var_vs = np.asarray([var_xy,var_xy,var_z])
    mean = [0,0,0] #Each distribution has a native mean of 0.
    cov = np.multiply(var_vs,np.identity(3)) #distributions are assumed to be independent
    #mean_vz = np.sqrt(2*var/np.pi) #compute this from all-positive component of distribution

    #Additional values are computed so that tuples with negative vz can be discarded
    flag_array_full = False
    while not flag_array_full:
        output = np.random.multivariate_normal(mean,cov,int(np.round(n_part*3)))
        pos_output = output[np.where(output[:,2] > 0.)[0]]
        if pos_output.shape[0] >= n_part:
            flag_array_full = True
        
    return pos_output[:n_part]

def compute_cutoff_beta(T, frac=0.99):
    '''Returns the velocity for which the fraction frac of a beam emitted from a thermionic
    cathode with temperature T move more slowly in the longitudinal (z) direction.

    Arguments:
        T (float)               : temperature of the cathode in K
        frac (Optional[float])  : Fraction of beam with vz < cutoff. Defaults to 0.99.

    Returns:
        beta_cutoff (float)     : cutoff velocity divided by c.

    '''

    sigma = np.sqrt(2*kb_J*T/m) #effective sigma for half-Gaussian distribution

    multiplier = erfinv(frac) #the multiplier on sigma accounting for frac of the distribution

    return multiplier*sigma/c 
    


##########################################
###### INSTANTIATIONS
##########################################    

def constant_current(current, a0, b0):
    '''
    Instantiate a beam with constant, user-specified current and zero temperature.
    
    Arguments:
        current (float)         : beam current in Amperes
        a0      (float)         : X-plane source radius in m
        b0      (float)         : Y-plane source radius in m FOR 3D SIMULATIONS

    '''
    #top.inject = 1 must be specified in main script
    beam.ibeam  = current
    beam.a0     = a0
    beam.b0     = b0
    
    #fixed cathode temperature
    myInjector = injectors.injectorUserDefined(self,beam, 4.0, channel_width, z_part_min, ptcl_per_step)
    
    installuserinjection(myInjector.inject_electrons)
    
    # These must be set for user injection
    top.ainject = 1.0          
    top.binject = 1.0
    
    
    #cold beam approximation
    #beam.vthz   = 0
    #beam.vthperp= 0
    
    
#def child_langmuir_current(current,cathode_phi,anode_wf,grid_bias):
def child_langmuir_current(current, a0, b0):
    '''
    Instantiate a beam with (cold) Child-Langmuir limited current. Current must be computed using
    available geometry.
    
    Arguments:
        current (float): beam current in Amperes
        a0      (float): X-plane source radius in m
        b0      (float): Y-plane source radius in m FOR 3D SIMULATIONS
    '''
    
    #top.inject = 2 must be specified in main script
    beam.ibeam  = current
    beam.a0     = a0
    beam.b0     = b0
    #cold beam approximation 
    #beam.vthz   = 0
    #beam.vthperp= 0
    w3d.l_inj_exact = True #this is needed for top.inject=2
    
    
def thermionic_current(current, a0, b0, beam, cathode_temp, channel_width, z_part_min, ptcl_per_step):
    '''Instantiate a beam with the Richardson-Dushmann current and a thermal distribution of velocities.

    Arguments:
        current         (float) : beam current in Amperes
        a0              (float) : X-plane source radius in m
        b0              (float) : Y-plane source radius in m FOR 3D SIMULATIONS
        beam            (Warp)  : warp beam object (e.g. Species() call)
        cathode_temp    (float) : cathode temperature in K
        channel_width   (float) : width of domain in x/y plane
        z_part_min      (float) : z coordinate of particles injected - provides separation from left boundary
        ptcl_per_step   (int)   : number of macro particles injected per step        
    

    '''    
    #top.inject = 6 must be specified in main script
    
    beam.ibeam  = current
    beam.a0     = a0
    beam.b0     = b0


    myInjector = injectors.injectorUserDefined(self,beam, cathode_temp, channel_width, z_part_min, ptcl_per_step)
    
    installuserinjection(myInjector.inject_thermionic)
    
    # These must be set for user injection
    top.ainject = 1.0          
    top.binject = 1.0
        
    
    
    