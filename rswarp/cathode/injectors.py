"""
Classes for specifying different injectors for Warp simulations of cathodes.

Authors: Nathan Cook and Chris Hall
04/25/2017
"""

from __future__ import division
import numpy as np
import scipy
import sources


#Specify constants
from scipy.constants import e, m_e, c, k
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e #mass of electron




class injectorUserDefined(object):
    '''
    The injector class allows for user defined injections which require parameterization from user inputs. 
    Maintaining these 
    
    '''
    
    def __init__(self,beam,cathode_temp, channel_width, z_part_min, ptcl_per_step):
        '''
        Minimum class initialization for specifying particle coordinates.
        
        Arguments:
            beam                    : warp beam object (e.g. Species() call)
            cathode_temp (float)    : cathode temperature in K
            channel_width (float)   : width of domain in x/y plane
            z_part_min (float)      : z coordinate of particles injected - provides separation from left boundary
            ptcl_per_step (int)     : number of macro particles injected per step
        
        '''
        self.beam = beam
        self.cathode_temp = cathode_temp
        self.channel_width = channel_width
        self.z_part_min = z_part_min
        self.ptcl_per_step = ptcl_per_step
        #self.top = top
        
    def inject_thermionic(self):
        '''Define particle coordinates for thermionic injection. Note that this does not specify current, just macroparticle coordinates'''
        v_coords = sources.get_MB_velocities(self.ptcl_per_step,self.cathode_temp)
        x_vals = self.channel_width*(np.random.rand(self.ptcl_per_step)-0.5)
        y_vals = self.channel_width*(np.random.rand(self.ptcl_per_step)-0.5)
        z_vals = np.zeros(self.ptcl_per_step) + self.z_part_min #Add a minimum z coordinate to prevent absorption
        ptclArray = np.asarray([x_vals,v_coords[:,0],y_vals,v_coords[:,1],z_vals,v_coords[:,2]]).T
        self.beam.addparticles(x=ptclArray[:,0],y=ptclArray[:,2],z=ptclArray[:,4],
        vx=ptclArray[:,1],vy=ptclArray[:,3],vz=ptclArray[:,5])
        
    def inject_constant(self):
        '''Same as inject thermionic but with a very low default (4 K) temperature and no transverse velocities'''
        v_coords = sources.get_MB_velocities(self.ptcl_per_step,4)
        v_coords[:,0] = np.zeros(self.ptcl_per_step) #no transverse
        v_coords[:,1] = np.zeros(self.ptcl_per_step) #no transverse
        x_vals = self.channel_width*(np.random.rand(self.ptcl_per_step)-0.5)
        y_vals = self.channel_width*(np.random.rand(self.ptcl_per_step)-0.5)
        z_vals = np.zeros(self.ptcl_per_step) + self.z_part_min #Add a minimum z coordinate to prevent absorption
        ptclArray = np.asarray([x_vals,v_coords[:,0],y_vals,v_coords[:,1],z_vals,v_coords[:,2]]).T
        self.beam.addparticles(x=ptclArray[:,0],y=ptclArray[:,2],z=ptclArray[:,4],
        vx=ptclArray[:,1],vy=ptclArray[:,3],vz=ptclArray[:,5])
        
    def inject_thermionic_egun(self):
        '''
        Define particle coordinates for thermionic injection. Note that this does not specify current, just macroparticle coordinates.
        The "egun" mode modifies the injector call to adjust certain top quantities after a single particle addition.
        
        '''
        v_coords = sources.get_MB_velocities(self.ptcl_per_step,self.cathode_temp)
        x_vals = self.channel_width*(np.random.rand(self.ptcl_per_step)-0.5)
        y_vals = self.channel_width*(np.random.rand(self.ptcl_per_step)-0.5)
        z_vals = np.zeros(self.ptcl_per_step) + self.z_part_min #Add a minimum z coordinate to prevent absorption
        ptclArray = np.asarray([x_vals,v_coords[:,0],y_vals,v_coords[:,1],z_vals,v_coords[:,2]]).T
        #print "Ready to 'addparticles'"
        self.beam.addparticles(x=ptclArray[:,0],y=ptclArray[:,2],z=ptclArray[:,4],
        vx=ptclArray[:,1],vy=ptclArray[:,3],vz=ptclArray[:,5], lallindomain=True)        
        #print "Added particles"
        #self.top.inject = 100