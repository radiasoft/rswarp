import numpy as np
from scipy.constants import c as c0
from scipy.constants import e, m_e


def create_distribution(Npart, transverse_sigmas, length, z_sigma, seeds, symmetrize=False):
    """
    Create normally distributed partices in x, y and vx, vy, vz. Particle positions
    in z are created with spacing based on `length` / `Npart`.

    Args:
        Npart: (int) Number of macroparticles to create
        transverse_sigmas: (float)*4 rms sizes for: [x, y, vx, vy]
        length: (float) Total length of the distribution
        z_sigma: (float) rms size of vz
        seeds: (int)*6 seeds to use for initializing creation of each distribution
        symmetrize: (bool) If true then for each particle coordinate vector 'v' append
        '-v' to the distribution. Will result in creation of 2*`Npart` particles.

    Returns:

    """
    #

    x_rms_ini, y_rms_ini, vx_rms_ini, vy_rms_ini = transverse_sigmas
    z_max, z_min = length / 2., -length / 2.
    vz_rms_ini = z_sigma

    # Initialize RNG
    rng_seed_x, rng_seed_y, rng_seed_z, rng_seed_vx, rng_seed_vy, rng_seed_vz = seeds
    rand1 = np.random.RandomState(rng_seed_x)
    rand2 = np.random.RandomState(rng_seed_y)
    rand3 = np.random.RandomState(rng_seed_z)
    rand4 = np.random.RandomState(rng_seed_vx)
    rand5 = np.random.RandomState(rng_seed_vy)
    rand6 = np.random.RandomState(rng_seed_vz)

    # Create position data
    x = rand1.normal(0., x_rms_ini, Npart)
    x -= np.mean(x)
    x *= x_rms_ini / np.std(x)
    y = rand2.normal(0., y_rms_ini, Npart)
    y -= np.mean(y)
    y *= y_rms_ini / np.std(y)
    # z = rand3.uniform(z_min, z_max, Npart)
    dz = (z_max - z_min) / np.float64(Npart)
    z = np.linspace(z_min + 0.5 * dz, z_max - 0.5 * dz, Npart)

    # Create velocity data
    beam_x_minus = rand4.normal(0., vx_rms_ini, Npart)
    beam_x_minus -= np.mean(beam_x_minus)
    beam_x_minus *= vx_rms_ini / np.std(beam_x_minus)
#     beam_x_minus /= c0
    beam_y_minus = rand5.normal(0., vy_rms_ini, Npart)
    beam_y_minus -= np.mean(beam_y_minus)
    beam_y_minus *= vy_rms_ini / np.std(beam_y_minus)
#     beam_y_minus /= c0
    beam_z_minus = rand6.normal(0., vz_rms_ini, Npart)
    beam_z_minus -= np.mean(beam_z_minus)
    beam_z_minus *= vz_rms_ini / np.std(beam_z_minus)
#     beam_z_minus /= c0

    distribution = np.column_stack([x, y , z, beam_x_minus, beam_y_minus, beam_z_minus])
    if symmetrize:
        distribution = np.row_stack([distribution, -distribution])

    return distribution


def create_grid(lower_bounds, upper_bounds, cells):
    """
    Wrapper for NumPy meshgrid
    Args:
        lower_bounds: (float)*N lower bounds for the mesh in each dimension
        upper_bounds: (float)*N upper bounds for the mesh in each dimension
        cells: (float)*N number of cells along in each dimension

    Returns:
        (ndarray) * N
    """
    grids = []
    for lb, ub, c in zip(lower_bounds, upper_bounds, cells):
        grids.append(np.linspace(lb, ub, c))
    mesh = np.meshgrid(*grids)

    return mesh


def ion_electric_field(x, y, z, X_ion, charge=1, coreSq=1.0e-13):
    """
    Calculate the electric field components on a mesh in 3D Cartesian coordinates
    Args:
        x: (ndarray) x coordinates of the mesh
        y: (ndarray) y coordinates of the mesh
        z: (ndarray) z coordinates of the mesh
        X_ion: (float, float, float) x, y and z coordinates of the ion
        coreSq: (float) Softening parameter to remove singularity from the Coulomb interaction.
                default=1e-13 m**2

    Returns:
            ndarray, ndarray, ndarray: Ex, Ey, and Ez electric field components on the mesh
    """
    # Np = np.shape(x)[0]
    Ei_ion_x = x - X_ion[0]  # positive ion
    Ei_ion_y = y - X_ion[1]
    Ei_ion_z = z - X_ion[2]

    r3 = np.power(np.sqrt(Ei_ion_x * Ei_ion_x + Ei_ion_y * Ei_ion_y \
                          + Ei_ion_z * Ei_ion_z + coreSq), 3)
    Ei_ion_x = Ei_ion_x / r3[:] * charge
    Ei_ion_y = Ei_ion_y / r3[:] * charge
    Ei_ion_z = Ei_ion_z / r3[:] * charge
    # return Ei_ion  #  NB: un-normalized
    return Ei_ion_x, Ei_ion_y, Ei_ion_z


def drift_twiss(s, betas, alphas):
    # invariant in drift
    gammas = (1 + alphas**2) / betas

    # update
    alpha = alphas - s * gammas
    beta = betas - 2 * s * alphas + s**2 * gammas

    return alpha, beta, gammas


class DriftWeightUpdate:
    """
    Update Weights for Delta-f PIC
    Coded for electrons
    """

    def __init__(self, top, comm_world, species, gamma0, twiss, emittance, externally_defined_field=True):
        """
        Prepare to calculate weight updates for delta-f method
        For drift case the update method `update_weights` may be installed before step only
        Args:
            top: Warp's top object
            comm_world: Warp's MPI communicator object
            species: Warp species object for the electron beam
            gamma0: (float) Reference relativistic gamma for the beam frame
            twiss: (float)*4 Initial Twiss values in form (betax, alphax, betay, alphay)
            emittance: (float)*2 Initial emittance (un-normalized)
            externally_defined_field: (boolean) If false the exact ion fields are calculate from particle
            positions supplied by Warp. Otherwise the user should install an external electric field to
            Warp. In that case Warp is queried for the field at each particle location. This requires
            care in setting grid resolution.
        """
        self.top = top
        self.comm_world = comm_world
        self.mpi_size = comm_world.size
        self.species = species
        self.twiss = twiss
        self.gamma0 = gamma0
        self.beta0 =  np.sqrt(1. - 1. / (gamma0**2))
        self.emit_x, self.emit_y = emittance
        self.externally_defined_field = externally_defined_field
        self.softening_parameter = 1.0e-13  # Only used if externally_defined_field == False
        self.ion_velocity = np.array([0., 0., 0.])
        self._ion_position = np.array([0., 0., 0.])

    def update_weights(self):
        """
        To be called by a Warp install wrapper
        """
        # Needs to be before udpates for the step
        self._set_twiss_at_s()
        if np.any(np.abs(self.ion_velocity) > 0.):
            self._update_ion_position()

        dt = self.top.dt
        q2m = -e / m_e

        # Current particle quantities from Warp
        weights = self.top.pgroup.pid[:self.top.nplive, self.top.wpid - 1]
        x = self.top.pgroup.xp[:self.top.nplive]
        y = self.top.pgroup.yp[:self.top.nplive]
        z = self.top.pgroup.zp[:self.top.nplive]

        # Warp keeps gamma * v for the velocity component
        gamma_inv = self.top.pgroup.gaminv[:self.top.nplive]
        vx_n = self.top.pgroup.uxp[:self.top.nplive] * gamma_inv / c0 / (self.gamma0 * self.beta0)
        vy_n = self.top.pgroup.uyp[:self.top.nplive] * gamma_inv / c0 / (self.gamma0 * self.beta0)
        vz_n = self.top.pgroup.uzp[:self.top.nplive] * gamma_inv / c0
        
        if self.externally_defined_field:
            E_x = self.top.pgroup.ex[:self.top.nplive]
            E_y = self.top.pgroup.ey[:self.top.nplive]
            E_z = self.top.pgroup.ez[:self.top.nplive]
        else:
            E_x, E_y, E_z = ion_electric_field(x, y, z, self._ion_position, charge=79, coreSq=self.softening_parameter)
            E_x = 29.9792458 * np.abs(-1.6021766208e-19) * E_x
            E_y = 29.9792458 * np.abs(-1.6021766208e-19) * E_y
            E_z = 29.9792458 * np.abs(-1.6021766208e-19) * E_z

        # Weight before update will be needed in the middle of the update sequence
        weights_minus = weights.copy()

        # Start updates
        # x component update
        weights += dt * q2m * (1. - weights_minus) * \
                   (self.alphax * x + self.betax * vx_n) * E_x / \
                   (self.gamma0 * self.beta0 * self.emit_x)

        # y component update
        weights += dt * q2m * (1. - weights_minus) * \
                   (self.alphay * y + self.betay * vy_n) * E_y / \
                   (self.gamma0 * self.beta0 * self.emit_y)

        # z component update
        if self.mpi_size > 1:
            # Perform sums over local particles
            vz_local_size = vz_n.size
            vz_sum_local = np.sum(vz_n)
            vz_sq_sum_local = np.sum(vz_n * vz_n)
            local_data = np.array([vz_local_size, vz_sum_local, vz_sq_sum_local], dtype=float).reshape(1, 3)
            local_data = np.repeat(local_data, self.mpi_size, axis=0)
            all_data = np.zeros([self.mpi_size, 3])

            # Calculate global Avg. StD. from collected sums
            self.comm_world.Alltoall(local_data, all_data)
            vz_size = np.sum(all_data[:, 0])
            vz_mean = np.sum(all_data[:, 1]) / vz_size
            vz_sq_mean = np.sum(all_data[:, 2]) / vz_size
            vz_std = np.sqrt(vz_sq_mean - vz_mean * vz_mean)
        else:
            vz_mean = np.mean(vz_n)
            vz_std = np.std(vz_n)

        weights += dt * q2m * (1. - weights_minus) * \
                   (1. / (vz_std * vz_std)) * (vz_n - vz_mean) * E_z

    def _set_twiss_at_s(self):
        """
        Assign betax, alphax, betay, alphay based on current time step in Warp
        Currently calculates from Twiss at s=0
        Returns:

        """
        betax, alphax, betay, alphay = self.twiss

        # Time step is in the beam frame
        # Convert to distance in lab frame
        s = self.top.it * self.top.dt * self.beta0 * self.gamma0 * c0

        self.alphax, self.betax, _ = drift_twiss(s, betax, alphax)
        self.alphay, self.betay, _ = drift_twiss(s, betay, alphay)

    def _update_ion_position(self):
        # Non-relativistic position update of the ion in the beam frame
        # Only can be used if externally_defined_field == False

        dt = self.top.dt
        self._ion_position += self.ion_velocity * dt

        if np.any(self._ion_position < [self.top.xmmin, self.top.ymmin, self.top.zmmin]) or \
           np.any(self._ion_position > [self.top.xmmax, self.top.ymmax, self.top.zmmax]):

            print("WARNING: Ion is outside the simulation domain")





