import numpy as np
from scipy.constants import c as c0


def create_distribution(Npart, transverse_sigmas, length, z_sigma, seeds):
    """
    Create normally distributed partices in x, y and vx, vy, vz. Particle positions
    in z are created with spacing based on `length` / `Npart`.

    Args:
        Npart: (int) Number of macroparticles to create
        transverse_sigmas: (float)*4 rms sizes for: [x, y, vx, vy]
        length: (float) Total length of the distribution
        z_sigma: (float) rms size of vz
        seeds: (int)*6 seeds to use for initializing creation of each distribution

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
