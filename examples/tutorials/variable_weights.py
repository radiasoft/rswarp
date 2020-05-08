"""
Warp run script for demonstrating particle weight control.
See notebook interactive_weight_control.ipynb for the tutorial.
"""

import numpy as np
import h5py as h5
import os

import warp as wp
from warp.data_dumping.openpmd_diag import ParticleDiagnostic


from rswarp.utilities.file_utils import cleanupPrevious

diagDir = 'diags/hdf5'
diagFDir = ['diags/fields/magnetic', 'diags/fields/electric']

# Cleanup command if directories already exist
# Warp's HDF5 diagnostics will not overwrite existing files.
if wp.comm_world.rank == 0:
    cleanupPrevious(diagDir, diagFDir)

####################
# General Parameters
####################

# Simulation Steps
Nsteps = 2000

# Set the solver geometry to cylindrical
wp.w3d.solvergeom = wp.w3d.XZgeom

# Switches
particle_diagnostic_switch = True  # Record particle data periodically

# Dimensions
length = 0.1
width = length / 2  # m

##########################
# 3D Simulation Parameters
##########################

# Set cells 
wp.w3d.nx = 20
wp.w3d.ny = 20
wp.w3d.nz = 20

# Boundaries don't matter here (no field solve)
wp.w3d.bound0 = wp.neumann 
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# Set boundary dimensions
wp.w3d.xmmin = -width
wp.w3d.xmmax = width
wp.w3d.ymmin = -width
wp.w3d.ymmax = width
wp.w3d.zmmin = 0.0
wp.w3d.zmmax = length

# Set particle boundary conditions
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb

# Type of particle push. 2 includes tan correction to Boris push.
wp.top.ibpush = 2

dx = (wp.w3d.xmmax - wp.w3d.xmmin) / wp.w3d.nx
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz  # Because warp doesn't set w3d.dz until after solver instantiated
wp.top.dt = 1e-9  # Timestep set to cell crossing time with some padding

####################################
# Create Beam and Set its Parameters
####################################

beam = wp.Species(type=wp.Electron, name='Electron', lvariableweights=True)
beam.sw = 100

NP = wp.w3d.nx * wp.w3d.nz
x = np.linspace(wp.w3d.xmmin + wp.w3d.xmmin / wp.w3d.nx,
                wp.w3d.xmmax - wp.w3d.xmmax / wp.w3d.nx,
                wp.w3d.nx)
z = np.linspace(0 + 0.5 * wp.w3d.zmmax / wp.w3d.nz,
                wp.w3d.zmmax - 0.5 * wp.w3d.zmmax / wp.w3d.nz,
                wp.w3d.nz)
zeros = np.zeros_like(x)

def load_particles():
    beam.addparticles(x=x, y=zeros, z=z, vx=zeros, vy=zeros, vz=zeros)
wp.installparticleloader(load_particles)

# Wanted to turn off deposition, but doesn't seem to work 
# wp.top.pgroup.ldodepos[0] = 0 

##############################
# Install Background Field(s)
##############################
# External B-Field can be added by setting vector components at each cell
# ez = np.zeros([wp.w3d.nx, wp.w3d.ny, wp.w3d.nz])
# ez[:, :, :] = 1e6  # V/m
# z_start = wp.w3d.zmmin
# z_stop = wp.w3d.zmmax
# dx = (wp.w3d.xmmax - wp.w3d.xmmin) / wp.w3d.nx
# # Add B-Field to simulation
# wp.addnewegrd(z_start, z_stop,
#               xs=wp.w3d.xmmin / 2., dx=dx/2.,
#               ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin),
#               nx=wp.w3d.nx, ny=wp.w3d.ny, nz=wp.w3d.nz, ez=ez)

# Uniform B_z field
# bz = np.zeros([wp.w3d.nx, wp.w3d.ny, wp.w3d.nz])
# bz[:, :, :] = 1.0
# z_start = wp.w3d.zmmin
# z_stop = wp.w3d.zmmax

# wp.addnewbgrd(z_start, z_stop, xs=wp.w3d.xmmin, dx=(wp.w3d.xmmax - wp.w3d.xmmin), ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin),
#            nx=wp.w3d.nx, ny=wp.w3d.ny, nz=wp.w3d.nz, bz=bz)

########################
# Register Field Solvers
########################

# prevent GIST plotting from starting upon setup
wp.top.lprntpara = False
wp.top.lpsplots = False


solverE = wp.MultiGrid2D()
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




###########################
# Generate and Run PIC Code
###########################

wp.derivqty()
wp.package("w3d")
wp.generate()

# wp.top.pgroup.pid[:NP, wp.top.wpid - 1] = 100.

# while wp.top.it < Nsteps:
#     wp.step(100)