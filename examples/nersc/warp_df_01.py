"""
Testing of delta-f implementation in Warp for a drift
"""

import numpy as np
import warp as wp
import sys, os
base_directory = os.environ['SCRATCH']
sys.path.insert(1, os.path.join(base_directory, 'rswarp/rswarp/run_files/delta_f/'))
import delta_f_tools as dft

sys.path.insert(1,  os.path.join(base_directory, 'rswarp/rswarp/diagnostics/'))
from particle_diag import ParticleDiagnostic

from rswarp.utilities.file_utils import cleanupPrevious
diagDir = 'diags/hdf5'
diagFDir = [os.path.join(base_directory, 'diags/fields/magnetic'), os.path.join(base_directory,'diags/fields/electric')]

# Cleanup command if directories already exist
# Warp's HDF5 diagnostics will not overwrite existing files.
if wp.comm_world.rank == 0:
    cleanupPrevious(diagDir, diagFDir)
if wp.comm_world.size != 1:
    wp.synchronizeQueuedOutput_mpi4py(out=True, error=False)

####################
# Simulation Setup
####################

Npart = 65536 * 120
Nstep_tw = 3700  # number of loc-ns at which Courant-Snyder params are computed (interp-n in between)

gamma0 = 42.66  # assumed exact
beta0 = np.sqrt(1. - 1. / (gamma0 * gamma0))
sigma_gamma_over_gamma = 1.0e-3  # rms energy spread in the lab frame

L_mod = 3.7  # m, modulator section length in the lab frame

T_mod = L_mod / (gamma0 * beta0 * wp.clight)  # sim time in the _beam_ frame

quad_grad = np.array([0.0, 0.0, 0.0, 0.0])  # quad field gradients (assuming 4 quads)

# ION PARAMETERS
Z_ion = 79
X_ion = np.array([0.0, 0.0, 0.0])
# V_ion = np.array([0.0, 0.0, 0.0])
coreSq = 1.0e-13  # m^2, a short-range softening parameter for the Coulomb potential: r^2 -> r^2 + coreSq

# ELECTRON BEAM PARAMETERS
I_el = 100.  # A, e-beam current
z_min_lab = -10.0e-5  # m
z_max_lab = 10.0e-5  # m

z_min = gamma0 * z_min_lab  # in the beam frame
z_max = gamma0 * z_max_lab  # beam frame

# Initial Courant-Snyder parameters (beam at the waist initially):
alpha_x_ini = 0.0
alpha_y_ini = 0.0
beta_x_ini = 4.5  # m
beta_y_ini = 4.5  # m
gamma_x_ini = (1. + alpha_x_ini * alpha_x_ini) / beta_x_ini
gamma_y_ini = (1. + alpha_y_ini * alpha_y_ini) / beta_y_ini

eps_n_rms_x = 5.0e-6  # m-rad, normalized rms emittance
eps_n_rms_y = 5.0e-6  # m-rad, normalized rms emittance
eps_rms_x = eps_n_rms_x / (gamma0 * beta0)
eps_rms_y = eps_n_rms_y / (gamma0 * beta0)

x_rms_ini = np.sqrt(eps_rms_x * beta_x_ini)
y_rms_ini = np.sqrt(eps_rms_y * beta_y_ini)
xp_rms_ini = np.sqrt(eps_rms_x * gamma_x_ini)  # in the lab frame, of course
yp_rms_ini = np.sqrt(eps_rms_y * gamma_y_ini)
vx_rms_ini = gamma0 * beta0 * wp.clight * xp_rms_ini  # in the _beam_ frame
vy_rms_ini = gamma0 * beta0 * wp.clight * yp_rms_ini  # in the _beam_ frame

vz_rms_ini = beta0 * wp.clight * sigma_gamma_over_gamma  # in the _beam_ frame


####################
# General Parameters
####################

# Simulation Steps
Nsteps = 5000


# Set the solver geometry to cylindrical
wp.w3d.solvergeom = wp.w3d.XYZgeom

# Switches
particle_diagnostic_switch = True  # Record particle data periodically


##########################
# 3D Simulation Parameters
##########################

# Set cells
wp.w3d.nx = 32
wp.w3d.ny = 32
wp.w3d.nz = 32

# Boundaries don't matter here (no field solve)
wp.w3d.bound0 = wp.neumann
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# calculate transverse domain size we will need at the end
# TODO: This could be made more rigorous for this particular example
#  so that we guarantee there is periodic shifting

_, b, _ = dft.drift_twiss(L_mod, beta_x_ini, alpha_x_ini)
width = 8 * np.sqrt(b * eps_rms_x)  # factor of 8 may not be enough with a gaussian with no cutoff

# Set boundary dimensions
wp.w3d.xmmin = -width
wp.w3d.xmmax = width
wp.w3d.ymmin = -width
wp.w3d.ymmax = width
wp.w3d.zmmin = z_min * 1.1
wp.w3d.zmmax = z_max * 1.1

# Set particle boundary conditions
wp.top.pbound0 = wp.periodic
wp.top.pboundnz = wp.periodic

# Type of particle push. 2 includes tan correction to Boris push.
wp.top.ibpush = 2

dx = (wp.w3d.xmmax - wp.w3d.xmmin) / wp.w3d.nx
dy = (wp.w3d.ymmax - wp.w3d.ymmin) / wp.w3d.ny
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

wp.top.dt = T_mod / np.float64(Nsteps)

####################################
# Create Beam and Set its Parameters
####################################
# separate RNG seeds for different phase space coords, so that increasing the number
# of particles from N1 to N2 > N1 we get the same initial N1 particles among the N2
seeds = [98765, 87654, 76543, 65432, 54321, 43210]

# If species weight is 0 and variable weights are set I think there will be no deposition
# Since weight is based on sw * weight_array
beam = wp.Species(type=wp.Electron, name='Electron', lvariableweights=True)
beam.sw = 0

transverse_sigmas = [x_rms_ini, y_rms_ini, vx_rms_ini, vy_rms_ini]
initial_distribution_f0 = dft.create_distribution(Npart=Npart, transverse_sigmas=transverse_sigmas,
                                                  length=z_max-z_min, z_sigma=vz_rms_ini, seeds=seeds,
                                                  symmetrize=True, four_fold=True)

def load_particles():
    beam.addparticles(x=initial_distribution_f0[:, 0],
                      y=initial_distribution_f0[:, 1],
                      z=initial_distribution_f0[:, 2],
                      vx=initial_distribution_f0[:, 3],
                      vy=initial_distribution_f0[:, 4],
                      vz=initial_distribution_f0[:, 5])


wp.installparticleloader(load_particles)


##############################
# Install Single Ion Field
##############################
# External B-Field can be added by setting vector components at each cell
# X, Y, Z = dft.create_grid((wp.w3d.xmmin, wp.w3d.ymmin, wp.w3d.zmmin),
#                           (wp.w3d.xmmax, wp.w3d.ymmax, wp.w3d.zmmax),
#                           (2*wp.w3d.nx, 2*wp.w3d.ny, 2*wp.w3d.nz))

# ex, ey, ez = dft.ion_electric_field(X, Y, Z, X_ion, charge=Z_ion, coreSq=0.)
# ex = 29.9792458 * np.abs(-1.6021766208e-19) * ex
# ey = 29.9792458 * np.abs(-1.6021766208e-19) * ey
# ez = 29.9792458 * np.abs(-1.6021766208e-19) * ez

# z_start = wp.w3d.zmmin
# z_stop = wp.w3d.zmmax

# # Add B-Field to simulation
# wp.addnewegrd(z_start, z_stop,
#               xs=wp.w3d.xmmin, dx=dx/2.,
#               ys=wp.w3d.ymmin, dy=dy/2., 
#               nx=2*wp.w3d.nx, ny=2*wp.w3d.ny, 
#               nz=2*wp.w3d.nz,
#               ex=ex, ey=ey, ez=ez)


########################
# Register Field Solvers
########################

# prevent GIST plotting from starting upon setup
wp.top.lprntpara = False
wp.top.lpsplots = False


solverE = wp.MultiGrid3D()
solverE.mgtol = 1e-2  # TODO: right now we don't account for space charge
wp.registersolver(solverE)
solverE.mgverbose = -1
wp.top.verbosity = -1

######################
# Particle Diagnostics
######################

# HDF5 Particle/Field diagnostic options

if particle_diagnostic_switch:
    particleperiod = 1
    particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=wp.top, w3d=wp.w3d,
                                               # Include data from all existing species in write
                                               species={species.name: species for species in wp.listofallspecies},
                                               comm_world=wp.comm_world, lparallel_output=False,
                                               # `ParticleDiagnostic` automatically append 'hdf5' to path name
                                               write_dir=diagDir[:-5])
    wp.installafterstep(particle_diagnostic_0.write)  # Write method is installed as an after-step action

    
#######################
# Install Weight Update
#######################


twiss_init = (beta_x_ini, alpha_x_ini, beta_y_ini, alpha_y_ini)
emit_init = (eps_rms_x, eps_rms_y)
weight_update = dft.DriftWeightUpdate(wp.top, wp.comm_world, beam, gamma0, twiss_init, emit_init, externally_defined_field=False)
wp.installbeforestep(weight_update.update_weights)

###########################
# Generate and Run PIC Code
###########################

wp.derivqty()
wp.package("w3d")
wp.generate()

# Initialize weights to 0 for delta-f algorithm
try:
    # Longitudinal domain is larger than the beam so no all processes start with particle arrays initialized
    wp.top.pgroup.pid[:wp.top.nplive, wp.top.wpid - 1] *= 0
except:
    pass

wp.step(1)
particle_diagnostic_0.period = 250
wp.step(Nsteps - 1)

rank = wp.comm_world.rank
np.save('diags/final_weights_{}.npy'.format(rank), [wp.top.pgroup.xp[:wp.top.nplive], wp.top.pgroup.yp[:wp.top.nplive], wp.top.pgroup.zp[:wp.top.nplive], 
        wp.top.pgroup.pid[:wp.top.nplive, wp.top.wpid - 1]])
