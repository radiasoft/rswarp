"""
 Simulates a continuously injected 3.26 MeV electron beam streaming through a background H2 with impact ionization
 to create space charge neutralization.

 This version of magnetized_cooler is being ported to PICMI.
 The following features are removed currently:
    - ionization
    - conductors
    - Background B-Field
    - ZCrossing diagnostics
"""
import numpy as np
from warp import picmi
cst = picmi.constants

# from rswarp.cathode.injectors import UserInjectors

# Seed set for reproduction
np.random.seed(967644)


####################
# General Parameters
####################

# Simulation Steps
Nsteps = 2000

# Set the solver geometry to cylindrical
# Geometry set by CylindricalGrid

# Switches
particle_diagnostic_switch = True  # Record particle data periodically
field_diagnostic_switch = True  # Record field/potential data periodically
user_injection = True  # Switches injection type
space_charge = True  # Controls field solve on/off
simulateIonization = True  # Include ionization in simulation
installConductors = True  # Setup electrode `Conductors`

if user_injection:
    # User injection thermionic_rz_injector method uses a r**2 scaling to distribute particles uniformly
    variable_weight = False
else:
    # Warp default injection uses radially weighted particles to make injection uniform in r
    variable_weight = True

# Dimensions for the electron cooler
pipe_radius = 0.03
cooler_length = 30.  # m
cathode_temperature = 0.25  # eV

# Beam
beam_beta = 0.990813945176
beam_ke = cst.emass / cst.jperev * cst.clight ** 2 * (1. / np.sqrt(1 - beam_beta ** 2) - 1.)  # eV
beam_current = 3.  # A
beam_radius = 0.01  # m

##########################
# 3D Simulation Parameters
##########################

# Set cells (nx == ny in cylindrical coordinates)
nx = 40
ny = 40
nz = 40000

# Set field boundary conditions (Warp only allows setting x and y together)
bound0 = 'neumann'  # Use neumann to remove field from electrode edge at boundary
boundnz = 'neumann'
boundxy = 'neumann'

# Set boundary dimensions
xmmin = -pipe_radius
xmmax = pipe_radius
ymmin = -pipe_radius
ymmax = pipe_radius
zmmin = 0.0
zmmax = cooler_length

# TODO: Where are pbound conditions set?
# pbound0 = wp.absorb
# pboundnz = wp.absorb

dx = (xmmax - xmmin) / nx
dz = (zmmax - zmmin) / nz  # Because warp doesn't set w3d.dz until after solver instantiated
dt = 0.75 * dz / (beam_beta * cst.clight)  # Timestep set to cell crossing time with some padding

grid = picmi.CylindricalGrid(
        number_of_cells           = [nx, nz],
        lower_bound               = [zmmin, zmmax],
        upper_bound               = [xmmin, xmmax],
        lower_boundary_conditions = [boundxy, bound0],
        upper_boundary_conditions = [boundxy, boundnz])

####################################
# Create Beam and Set its Parameters
####################################
# parameters for electron beam:
#   total energy: 3.2676813658 MeV
#   beta_e: 0.990813945176

ptcl_per_step = 400  # number of particles to inject on each step

if space_charge:
    beam_weight = beam_current * dt * ptcl_per_step / cst.e
else:
    ibeam = 0.0

# wp.w3d.l_inj_exact = True

"""
A custom injector routine is used here to allow for a very relativistic beam to be injected directly into the simulation
because Warp's built in routine is not based on relativistic kinematics. Setting user_injection = False should work
well for lower energy beams.
"""

# beam_dist = picmi.GaussianBunchDistribution(
#             n_physical_particles = bunch_physical_particles,
#             rms_bunch_size       = bunch_rms_size,
#             rms_velocity         = bunch_rms_velocity,
#             centroid_position    = bunch_centroid_position,
#             centroid_velocity    = bunch_centroid_velocity )
# beam = picmi.Species(particle_type='electron',
#                      name='beam',
#                      initial_distribution=beam_dist)
if user_injection:
    print("No injection included. Not sure if continuous injection is supported.")

##############################
# Ionization of background gas
##############################

# Ionization e- + H2 -> H2+ + e- + e-

if simulateIonization is True:
    print("Custom Ionization physics not currently included")

####################
# Install Conductors
####################
# Create conductors that will represent electrodes placed inside vacuum vessel

if installConductors is True:
    print("Internal conductors not currently included")

##############################
# Install Ideal Solenoid Field
##############################
# electron cooler interaction takes place inside a solenoid
# for time being this is represented as an ideal magnetic field B = (0, 0, bz)

print('Background fields not implemented. Solenoid field not included')

# External B-Field can be added by setting vector components at each cell
# bz = np.zeros([wp.w3d.nx, wp.w3d.ny, wp.w3d.nz])
# bz[:, :, :] = 1.0  # T
# z_start = wp.w3d.zmmin
# z_stop = wp.w3d.zmmax
#
# # Add B-Field to simulation
# wp.addnewbgrd(z_start, z_stop,
#               xs=wp.w3d.xmmin, dx=(wp.w3d.xmmax - wp.w3d.xmmin),
#               ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin),
#               nx=wp.w3d.nx, ny=wp.w3d.ny, nz=wp.w3d.nz, bz=bz)

########################
# Register Field Solvers
########################

# prevent GIST plotting from starting upon setup
# wp.top.lprntpara = False
# wp.top.lpsplots = False

if space_charge:
    # magnetostatic solver disabled for now, was causing mpi issues and is very slow
    # solverB = wp.MagnetostaticMG()
    # solverB.mgtol = [0.01] * 3
    # wp.registersolver(solverB)

    # Add 2D Field Solve, will be cylindrical solver based on setting of w3d.solvergeom
    solverE = picmi.PICMI_ElectrostaticSolver(grid=grid,
                                              method='Multigrid')

# Conductors must be registered after Field Solver is instantiated if we want them to impact field solve

# Scraper setup

######################
# Particle Diagnostics
######################

# HDF5 Particle/Field diagnostic options
field_diag = picmi.FieldDiagnostic(grid = grid,
                                    period = 100,
                                    warpx_plot_raw_fields = 1,
                                    warpx_plot_raw_fields_guards = 1,
                                    warpx_plot_finepatch = 1,
                                    warpx_plot_crsepatch = 1)


if particle_diagnostic_switch:
    particleperiod = 50
    particle_diagnostic_0 = picmi.ParticleDiagnostic(period=100,
                                                     species=[])
print('No species given to particle diagnostic')

# particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=wp.top, w3d=wp.w3d,
#                                            # Include data from all existing species in write
#                                            species={species.name: species for species in wp.listofallspecies},
#                                            comm_world=wp.comm_world, lparallel_output=False,
#                                            # `ParticleDiagnostic` automatically append 'hdf5' to path name
#                                            write_dir=diagDir[:-5])


if field_diagnostic_switch:
    fieldperiod = 500
    field_diagnostic_0 = picmi.FieldDiagnostic(grid=grid,
                                               period=fieldperiod)

# Crossing Diagnostics
print("Zcrossing Diagnostics not implemented")

###########################
# Generate and Run PIC Code
###########################
sim = picmi.Simulation(solver=solverE,
                       time_step_size=dt,
                       verbose=1)

# Diagnostics
sim.add_diagnostic(particle_diagnostic_0)
sim.add_diagnostic(field_diagnostic_0)


print('Is there access to timestep information while running?')
# while wp.top.it < Nsteps:
#     sim.step(max_steps)

sim.step(Nsteps)

# ZCrossing Output
