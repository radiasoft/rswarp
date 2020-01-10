
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

class spheroid(protrusion):
    """
    Subclass for describing emission from a hemi spheroid emitter using variable weight macroparticles in Warp.

    spheroid overrides the base class function `compute_emission_sites`

    """

    def __init__(self, x_radius = 1.0e-7, z_radius = 1.0e-7,
                work_function = 2.05, temperature = 1000,
                x_offset = 0.0, y_offset = 0.0, voltage = 0.0, N_particles = 100,
                output_to_file = False, file_name = 'emitted_particles_sph.h5' ):

        '''
        spheroid class __init__ method. Follows base class call signature.

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

        self.x_radius = x_radius
        self.z_radius = z_radius

        z_pts = np.linspace(0, self.z_radius, 100)
        r_pts = self.x_radius * np.sqrt((self.z_radius ** 2.- z_pts ** 2.) / self.z_radius ** 2.)
        z_pts = np.append(z_pts, 0)
        r_pts = np.append(r_pts, 0)

        self.z_pts = z_pts
        self.r_pts = r_pts

        if self.x_radius < self.z_radius:
            self.area = np.pi * self.x_radius ** 2. + np.pi * self.x_radius * self.z_radius / (np.sqrt(self.z_radius ** 2. - self.x_radius **2)/self.z_radius) * np.arccos(self.x_radius / self.z_radius) / self.N_particles
        elif self.x_radius > self.z_radius:
            self.area = np.pi * self.x_radius ** 2. + np.pi * self.z_radius ** 2 / (np.sqrt(self.x_radius ** 2. - self.z_radius **2)/self.x_radius) * np.log((1 + (np.sqrt(self.x_radius ** 2. - self.z_radius **2)/self.x_radius)) / (self.z_radius / self.x_radius)) / self.N_particles
        else:
            self.area = self.x_radius ** 2. * 2. * np.pi / self.N_particles


    def compute_emission_sites(self, N_particles):
        '''Define emission sites for macroparticle injection specific to hemispherical emitter.'''
        if self.x_radius == self.z_radius:
            u = np.random.uniform(0, 1, N_particles)
            theta =  np.pi * u
            v = np.random.uniform(0, 1, N_particles)
            psi = np.arccos(2 * v - 1)

            y = np.ones(N_particles) * self.y_offset + self.x_radius * np.cos(theta) * np.sin(psi)
            z = self.x_radius * np.sin(theta) * np.sin(psi)
            x = np.ones(N_particles) * self.x_offset + self.x_radius * np.cos(psi)

            return x, y, z
        else:
            a = self.x_radius
            b = self.z_radius

            h = b * np.sqrt(np.random.uniform(0, 1, N_particles))
            r = self.x_radius * np.sqrt((self.z_radius ** 2.- h ** 2.) / self.z_radius ** 2.)
            t = 2 * np.pi * np.random.uniform(0, 1, N_particles)

            x = np.ones(N_particles) * self.x_offset + r * np.cos(t)
            y = np.ones(N_particles) * self.y_offset + r * np.sin(t)
            z =  h

            return x, y, z

    def install_conductor(self):
        protrusion.install_conductor(self)
        # To redefine, comment the above line and redefine below:

    def register(self, solver, beam, top):
        protrusion.register(self, solver, beam, top)
        # To redefine, comment the above line and redefine below:


class spherical(spheroid):
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

        spheroid.__init__(self, radius, radius, work_function, temperature,
                    x_offset, y_offset, voltage, N_particles,
                    output_to_file, file_name )


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

        z_pts = np.linspace(0, self.z_cone, 100)
        r_pts = - z_pts * self.r_cone / self.z_cone + self.r_cone
        z_pts = np.append(z_pts, 0)
        r_pts = np.append(r_pts, 0)

        self.z_pts = z_pts
        self.r_pts = r_pts

        self.area = self.r_cone * np.pi * np.sqrt(self.z_cone ** 2. + self.r_cone ** 2.)/ self.N_particles


    def compute_emission_sites(self, N_particles):
        '''Define emission sites for macroparticle injection specific to conical emitter.'''

        a = self.r_cone
        b = self.z_cone

        r = a * np.sqrt(np.random.uniform(0, 1, N_particles))
        h = (r / a) * b
        t = 2 * np.pi * np.random.uniform(0, 1, N_particles)

        x = np.ones(N_particles) * self.x_offset + r * np.cos(t)
        y = np.ones(N_particles) * self.y_offset + r * np.sin(t)
        z =  -h + self.z_cone

        return x, y, z

    def install_conductor(self):
        protrusion.install_conductor(self)
        # To redefine, comment the above line and redefine below:

    def register(self, solver, beam, top):
        protrusion.register(self, solver, beam, top)
        # To redefine, comment the above line and redefine below:



class gaussian(protrusion):

    def __init__(self, r_cutoff = 1.0e-7, a = 1.0e-7, sigma = 1.0e-7,
                work_function = 2.05, temperature = 1000,
                x_offset = 0.0, y_offset = 0.0, voltage = 0.0, N_particles = 100,
                output_to_file = False, file_name = 'emitted_particles_gauss.h5' ):
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

        self.r_cutoff = r_cutoff
        self.a = a
        self.sigma = sigma
        self.z_offset = a * np.exp( - (r_cutoff ** 2.) / (sigma ** 2. * 2))

        r_pts = np.linspace(0, r_cutoff, 100)
        z_pts = a * np.exp(- (r_pts ** 2.) / (2.0 * sigma ** 2.)) - self.z_offset
        z_pts = np.append(z_pts, 0)
        r_pts = np.append(r_pts, 0)

        self.r_pts = r_pts
        self.z_pts = z_pts

        self.area = self.compute_areas()

    def sec_gamma(self, X, Y):

        df_dx = - X * self.a / ( self.sigma ** 2.) * np.exp( - ( X ** 2. + Y ** 2.) / (2.0 * self.sigma ** 2.))
        df_dy = - Y * self.a / ( self.sigma ** 2.) * np.exp( - ( X ** 2. + Y ** 2.) / (2.0 * self.sigma ** 2.))

        return np.sqrt( 1 + df_dx ** 2. + df_dy ** 2.)

    def compute_areas(self):

        r_beam = self.r_cutoff
        n_sites = int(np.round(np.sqrt(self.N_particles)))
        n_r = n_sites
        n_th = n_sites

        r_bins = np.sqrt(np.linspace(self.r_cutoff**2./10000., self.r_cutoff ** 2., n_r))

        self.r_bins = r_bins
        self.r_beam = r_beam
        self.n_r = n_r
        self.n_th = n_th

        theta_bins = np.linspace(0, 2.0 * np.pi, n_th)
        r_centers = r_bins[0:-1] + np.diff(r_bins) / 2.
        theta_centers = theta_bins + np.mean(np.diff(theta_bins)) / 2.

        R_b, Th_b = np.meshgrid(r_bins, theta_bins)
        self.R_bins = R_b
        self.Th_bins = Th_b

        areas = (R_b[:,1::]**2. - R_b[:,0::-1] **2.) * np.pi * np.mean(np.diff(theta_bins)) / (2.0 * np.pi)

        X_b = R_b * np.cos(Th_b)
        Y_b = R_b * np.sin(Th_b)

        R_c, Th_c = np.meshgrid(r_centers, theta_centers)
        X_c = R_c * np.cos(Th_c)
        Y_c = R_c * np.sin(Th_c)

        x_cent = X_c.flatten()
        y_cent = Y_c.flatten()

        secgamma = self.sec_gamma(x_cent, y_cent)

        area = areas.flatten() * secgamma

        return area



    def compute_emission_sites(self, N_particles):
        '''Define emission sites for macroparticle injection specific to conical emitter.'''

        R_low = self.R_bins[:,0:-1]
        R_high = self.R_bins[:,1::]
        Th_low = self.Th_bins[0:-1,:]
        Th_high = self.Th_bins[1::,:]

        R_p = np.random.uniform(R_low, R_high)
        Th_p = np.random.uniform(Th_low, Th_high)

        r_part = R_p.flatten()
        th_part = Th_p.flatten()
        x = r_part * np.cos(th_part)
        y = r_part * np.sin(th_part)
        z = self.a * np.exp(- (x**2. + y**2.) / (self.sigma ** 2. * 2.)) - self.z_offset

        return x, y, z

    def install_conductor(self):
        protrusion.install_conductor(self)
        # To redefine, comment the above line and redefine below:

    def register(self, solver, beam, top):
        protrusion.register(self, solver, beam, top)
        # To redefine, comment the above line and redefine below:
