"""
Testing of spectrometer beamline implemented into Warp
"""

import numpy as np
import warp as wp
# import rswarp.run_files.delta_f.delta_f_tools as dft

from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from rswarp.utilities.file_utils import cleanupPrevious

diagDir = 'diags/hdf5'
diagFDir = ['diags/fields/magnetic', 'diags/fields/electric']

# Cleanup command if directories already exist
# Warp's HDF5 diagnostics will not overwrite existing files.
if wp.comm_world.rank == 0:
    cleanupPrevious(diagDir, diagFDir)


####################
# Simulation Setup
####################
wp.top.lrelativ = 1
# wp.top.relativity = 1

path = 'opal_spectrometer/data/spectrometer_START_GAUSSIAN.dat'
init_distr = np.loadtxt(path, skiprows=2)

beta0 =  np.average(init_distr[:, 5] / np.sqrt(1 + init_distr[:, 5]**2))
gamma0 = 1. / np.sqrt(1 - beta0**2)

beam_length_lab = 160 * wp.um  # Approximate length, with a little padding

beamline_length = 3.5 # m, should be 3.472 meters in OPAL
T_total = beamline_length / (gamma0 * beta0 * wp.clight)  # sim time in the _beam_ frame
# T_total = beamline_length / (beta0 * wp.clight)  # sim time in the _lab_ frame

####################
# General Parameters
####################

# Simulation Steps
Nsteps = 6500

# Set the solver geometry to cylindrical
wp.w3d.solvergeom = wp.w3d.XYZgeom

# Switches
particle_diagnostic_switch = True  # Record particle data periodically


##########################
# 3D Simulation Parameters
##########################

# Set cells
wp.w3d.nx = 64
wp.w3d.ny = 64
wp.w3d.nz = 16

# Boundaries don't matter here (no field solve)
wp.w3d.bound0 = wp.neumann  # wp.neumann
wp.w3d.boundnz = wp.neumann  # wp.neumann
wp.w3d.boundxy = wp.dirichlet  # wp.neumann


width = 600*wp.um   # really half width

# Set boundary dimensions
wp.w3d.xmmin = -width
wp.w3d.xmmax = width
wp.w3d.ymmin = -width
wp.w3d.ymmax = width
# wp.w3d.zmmin = -beam_length_lab * gamma0  # length in beam frame
wp.w3d.zmmin = -beam_length_lab  # length in lab frame
wp.w3d.zmmax = 0.

# Set particle boundary conditions
wp.top.pbound0 = wp.periodic
wp.top.pboundnz = wp.periodic

# Type of particle push. 2 includes tan correction to Boris push.
wp.top.ibpush = 2

dx = (wp.w3d.xmmax - wp.w3d.xmmin) / wp.w3d.nx
dy = (wp.w3d.ymmax - wp.w3d.ymmin) / wp.w3d.ny
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz

wp.top.dt = T_total / np.float64(Nsteps)

####################################
# Create Beam and Set its Parameters
####################################
beam = wp.Species(type=wp.Electron, name='Electron', lvariableweights=True)
beam.sw = 1
beam.ibeam = 1.
# Beam velocity must be set manually here to be loaded into top
# Not set by derivqty otherwise because we use a addparticles call to set particles
beam.vbeam = wp.clight * beta0

# Convert from beta*gamma to velocity
init_distr[:, 1] = init_distr[:, 1] / np.sqrt(1 + init_distr[:, 1]**2) * wp.clight
init_distr[:, 3] = init_distr[:, 3] / np.sqrt(1 + init_distr[:, 3]**2) * wp.clight
init_distr[:, 5] = init_distr[:, 5] / np.sqrt(1 + init_distr[:, 5]**2) * wp.clight

def load_particles():
    beam.addparticles(x=init_distr[:, 0],
                      y=init_distr[:, 2],
                      z=init_distr[:, 4] - np.max(init_distr[:, 4]) * 1.05,  # We start head of bunch at z = 0
                      vx=init_distr[:, 1] / gamma0,
                      vy=init_distr[:, 3] / gamma0,
                      vz=init_distr[:, 5] - beta0 * wp.clight)# + np.ones_like(initial_distribution_f0[:, 5]) * beta0 * wp.clight)


wp.installparticleloader(load_particles)
wp.derivqty()

##############################
# Install Single Ion Field
##############################

########################
# Register Field Solvers
########################
# Solver setup handled by Multigrid3D
wp.top.fstype = -1

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

    
#################
# Install lattice
#################
# beamline_1: LINE=("drift_und2quads#0","q_def#0","drift_quad2quad#0",
# "q_foc#0","drift_quad2dip#0","spectr_dipole#0","drift_dip2dump#0");

opal_k1 = 21.808275280025114
angle=0.5235987755982988
dip_l=0.49430796473268457
dip_rc = dip_l / angle

drift_und2quads = wp.Drft(l=0.1)  # TODO: Changed from 1.0 m to get to quad quicker during testing
q_def = wp.Quad(l=0.074, db=opal_k1)
drift_quad2quad = wp.Drft(l=1.074)
q_foc = wp.Quad(l=0.074, db=-opal_k1)
drift_quad2dip = wp.Drft(l=1.5)
spectr_dipole = wp.Bend(l=dip_l, rc=dip_rc)
drift_dip2dump = wp.Drft(l=0.3)

beamline = drift_und2quads + q_def + drift_quad2quad + q_foc + drift_quad2dip + spectr_dipole + drift_dip2dump
wp.madtowarp(beamline)

###########################
# Generate and Run PIC Code
###########################
# TODO: top.depos temporarily turned off during generate call in HlF_strip (ln 245)

wp.package("w3d")
wp.generate()

# Quantities of interest
print('Beam velocity', wp.top.vbeam)
print('Frame velocity', wp.top.vbeamfrm)
print('Number of Steps',Nsteps)
print('Time step size', wp.top.dt)

n_step_taken = 0
step_size = 100

while n_step_taken < Nsteps:
    wp.step(step_size)
    n_step_taken += step_size
    print('Average z from getz', np.average(beam.getz()))
    print('Average z * gamma', np.average(beam.getz()) * gamma0)
    print('RMS sizes (um): ', np.std(beam.getx()) * 1e6, np.std(beam.gety()) * 1e6, np.std(beam.getz()) * 1e6)
    print('Live particles', beam.getz().size)
    print('Magnetic Field:', beam.getbx()[:20])
    print('')

# Initialize weights to 0 for delta-f algorithm
# wp.top.pgroup.pid[:wp.top.nplive, wp.top.wpid - 1] *= 0
# print(wp.top.zbeam)
# wp.step(1)
# print(wp.top.zbeam)
# particle_diagnostic_0.period = 250
# wp.step(Nsteps - 1)
# print(wp.top.zbeam)
# print('lab frame location?', wp.top.hzbeam)
# np.save('diags/final_weights.npy', wp.top.pgroup.pid[:wp.top.nplive, wp.top.wpid - 1])