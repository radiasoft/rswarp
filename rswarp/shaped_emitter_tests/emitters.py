
from __future__ import division
import h5py as h5
from warp import *

# Constants imports
from scipy.constants import e, m_e, c, k, epsilon_0, elementary_charge

q = elementary_charge
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e


class protrusion():
    """Base class for describing emission from shaped emitters using variable weight macroparticles in Warp.

    Instantiating the class requires specification of bulk emission properties, as well as the set of points describing
    the emission surface. The `compute_emission_sites` method then generates a distribution of locations for emitting
    macroparticles. These macroparticles have variable weights, and the `injector` method computes the effective weights
    for each particle based upon the current density along the emission surface. Subclasses should be developed to
    override the `compute_emission_sites` method with geometry-specific alterations.

    The `dump` method produces an hdf5 dataset containing particle phase space information.

    The `register` and `install_conductor` methods reproduce Warp's own functions of the same name.

    """


    def __init__(self, work_function = 2.05, temperature = 1000,
                x_offset = 0.0, y_offset = 0.0, voltage = 0.0, N_particles = 100,
                output_to_file = False, file_name = 'emitted_particles_protrusion.h5' ):

        '''
        Base class __init__ method for protrusion class.

        Args:
            work_function (float, optional)     : Emitter workfunction in eV. Defaults to 2.05 eV.
            temperature (float, optional)       : Emitter temperature in K. Defaults to 1000 K.
            x_offset, y_offset (float, optional):  Defaults to 0.
            voltage (float, optional)           : Potential at emitter surface. Defaults to 0.
            N_particles (int, optional)         : Number of macroparticles to be injected each step. Defaults to 100.
            output_to_file (Bool, optional)     : Flag for dumping particle phase spaces. Defaults to `emitted_
                                                  particles_protrusion.h5`

        '''

        self.work_function = work_function
        self.temperature = temperature
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.voltage = voltage
        self.N_particles = N_particles
        self.output_to_file = output_to_file
        self.file_name = file_name

        if output_to_file:
            self.hf = h5.File(self.file_name, 'w')
            self.index = 0

        #subclasses should override these values for specific geometries
        self.r_pts = np.linspace(0, 0, 100)
        self.z_pts = self.r_pts
        self.area = 0


    def compute_emission_sites(self, N_particles):
        '''Define emission sites for macroparticle injection. Randomized for base protrusion class.'''

        x = np.random.uniform(0, 1, N_particles)
        y = np.random.uniform(0, 1, N_particles)
        z = np.random.uniform(0, 1, N_particles)

        return x, y, z


    def dump(self, x, y, z, vx, vy, vz, Ex, Ey, Ez, phi, w):
        '''Dump particle phase spaces to hdf5 for testing.'''

        index = self.index
        data_set_name = 'step_' + str(index).zfill(4)

        array_data = np.column_stack([x, y, z, vx, vy, vz, Ex, Ey, Ez, phi, w])

        self.hf.create_dataset(data_set_name, data = array_data)

        self.index += 1

        return

    def register(self, solver, beam, top):
        '''Identifies the solver and other Warp objects needed by the conductor'''

        self.solver = solver
        self.beam = beam
        self.top = top
        self.beam.sw = 1


    def install_conductor(self):
        '''Installs the conductor in the Warp simulation'''

        self.conductor = ZSrfrv(rsrf = self.r_pts, zsrf = self.z_pts,
                        xcent = self.x_offset, ycent = self.y_offset,
                        zcent = 0, voltage = self.voltage)

        self.solver.installconductor(self.conductor, dfill=largepos)

        return


    def injector(self):
        '''Defines the particle injection using Schottky-Corrected Thermionic Emission.'''

        x, y, z = self.compute_emission_sites(self.N_particles)
        vx = np.zeros(len(x))
        vy = np.zeros(len(x))
        vz = np.zeros(len(x))

        Ex_s = fzeros(x.shape)
        Ey_s = fzeros(x.shape)
        Ez_s = fzeros(x.shape)
        Bx_s = fzeros(x.shape)
        By_s = fzeros(x.shape)
        Bz_s = fzeros(x.shape)
        phi_s = fzeros(x.shape)

        self.solver.fetchfieldfrompositions(x, y, z,
                                       Ex_s, Ey_s, Ez_s,
                                       Bx_s, By_s, Bz_s)

        self.solver.fetchpotentialfrompositions(x, y, z, phi_s)

        E_tot = np.sqrt(Ex_s ** 2. + Ey_s ** 2. + Ez_s ** 2.)

        dW = np.sqrt(E_tot * (q ** 3.) / (4. * np.pi * epsilon_0))
        w = self.work_function * q - dW

        J = 1.2e6 * self.temperature ** 2. * np.exp(- w / (k * self.temperature))

        I_site = J * self.area

        weights = I_site * self.top.dt / q

        self.beam.addparticles(x = x,
                      y = y,
                      z = z,
                      vx = vx,
                      vy = vy,
                      vz = vz,
                      w = weights)

        if self.output_to_file:
            self.dump(x, y, z, vx, vy, vz, Ex_s, Ey_s, Ez_s, phi_s, weights)

class spherical(protrusion):
    """
    Subclass for describing emission from a hemispherical emitter using variable weight macroparticles in Warp.

    spherical overrides the base class function `compute_emission_sites`

    """

    def __init__(self, radius = 1.0e-7,
                work_function = 2.05, temperature = 1000,
                x_offset = 0.0, y_offset = 0.0, voltage = 0.0, N_particles = 100,
                output_to_file = False, file_name = 'emitted_particles_sph.h5' ):

        '''
        spherical class __init__ method. Follows base class call signature.

        Args:
            work_function (float, optional)     : Emitter workfunction in eV. Defaults to 2.05 eV.
            temperature (float, optional)       : Emitter temperature in K. Defaults to 1000 K.
            x_offset, y_offset (float, optional):  Defaults to 0.
            voltage (float, optional)           : Potential at emitter surface. Defaults to 0.
            N_particles (int, optional)         : Number of macroparticles to be injected each step. Defaults to 100.
            output_to_file (Bool, optional)     : Flag for dumping particle phase spaces. Defaults to `emitted_
                                                  particles_sph.h5`

        '''


        protrusion.__init__(self,work_function, temperature,
                    x_offset, y_offset, voltage, N_particles,
                    output_to_file, file_name )

        self.radius = radius

        self.z_pts = np.linspace(0, self.radius, 100)
        self.r_pts = np.sqrt(self.radius ** 2. - self.z_pts ** 2.)
        self.area = self.radius ** 2. * 4. * np.pi / self.N_particles


    def compute_emission_sites(self, N_particles):
        '''Define emission sites for macroparticle injection specific to hemispherical emitter.'''

        u = np.random.uniform(0, 1, N_particles)
        theta =  np.pi * u
        v = np.random.uniform(0, 1, N_particles)
        psi = np.arccos(2 * v - 1)

        y = self.radius * np.cos(theta) * np.sin(psi)
        z = self.radius * np.sin(theta) * np.sin(psi)
        x = self.radius * np.cos(psi)

        return x, y, z

    def install_conductor(self):
        protrusion.install_conductor(self)
        # To redefine, comment the above line and redefine below:

    def register(self, solver, beam, top):
        protrusion.register(self, solver, beam, top)
        # To redefine, comment the above line and redefine below:

class conical(protrusion):
    """Subclass for describing emission from a conical emitter using variable weight macroparticles in Warp."""

    def __init__(self, r_cone = 1.0e-7, z_cone = 1.0e-7,
                work_function = 2.05, temperature = 1000,
                x_offset = 0.0, y_offset = 0.0, voltage = 0.0, N_particles = 100,
                output_to_file = False, file_name = 'emitted_particles_con.h5' ):
        '''
        conical class __init__ method. Follows base class call signature.

        Args:
            work_function (float, optional)     : Emitter workfunction in eV. Defaults to 2.05 eV.
            temperature (float, optional)       : Emitter temperature in K. Defaults to 1000 K.
            x_offset, y_offset (float, optional):  Defaults to 0.
            voltage (float, optional)           : Potential at emitter surface. Defaults to 0.
            N_particles (int, optional)         : Number of macroparticles to be injected each step. Defaults to 100.
            output_to_file (Bool, optional)     : Flag for dumping particle phase spaces. Defaults to `emitted_
                                                  particles_sph.h5`

        '''


        protrusion.__init__(self,work_function, temperature,
                    x_offset, y_offset, voltage, N_particles,
                    output_to_file, file_name )

        self.z_cone = z_cone
        self.r_cone = r_cone

        self.z_pts = np.linspace(0, self.z_cone, 100)
        self.r_pts = - self.z_pts * self.r_cone / self.z_cone + self.r_cone
        self.area = self.r_cone * np.pi * (self.r_cone + np.sqrt(self.z_cone ** 2. + self.r_cone ** 2.)) / self.N_particles


    def compute_emission_sites(self, N_particles):
        '''Define emission sites for macroparticle injection specific to conical emitter.'''

        a = self.r_cone
        b = self.r_cone
        h = self.z_cone

        h = a * np.sqrt(np.random.uniform(0, 1, N_particles))
        r = (b / a) * h
        t = 2 * np.pi * np.random.uniform(0, 1, N_particles)

        x = r * np.cos(t)
        y = r * np.sin(t)
        z =  -h + self.z_cone

        return x, y, z

    def install_conductor(self):
        protrusion.install_conductor(self)
        # To redefine, comment the above line and redefine below:

    def register(self, solver, beam, top):
        protrusion.register(self, solver, beam, top)
        # To redefine, comment the above line and redefine below:
