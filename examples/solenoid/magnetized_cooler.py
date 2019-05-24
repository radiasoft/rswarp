# this simulation runs 62% faster on 20 jupyter cores than on 10

from __future__ import division
import warp as wp
import numpy as np
import h5py as h5
import math
import os

from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.particles.singleparticle import TraceParticle
from warp.particles.extpart import ZCrossingParticles

# Use more-up-to-date local rswarp
import sys
sys.path.insert(1, '/home/vagrant/jupyter/rswarp')
from rswarp.diagnostics import FieldDiagnostic
from rswarp.cathode.injectors import UserInjectors
from rswarp.utilities.file_utils import cleanupPrevious

# Seed set for reproduction
np.random.seed(967644)

diagDir = 'diags/hdf5'
diagFDir = ['diags/fields/magnetic', 'diags/fields/electric']

# Cleanup command if directories already exist
# Warp's HDF5 diagnostics will not overwrite existing files. This command cleans the diagnostic directory
# to allow for rerunning the simulation
#if wp.comm_world.rank == 0:
#    cleanupPrevious(diagDir, diagFDir)
#    try:
#        os.remove('./diags/crossing_record.h5')
#    except OSError:
#        pass

####################
# General Parameters
####################
# Set the solver geometry to cylindrical
wp.w3d.solvergeom = wp.w3d.RZgeom

# Switches
particle_diagnostic_switch = True  # Record particle data periodically
field_diagnostic_switch = True  # Record field/potential data periodically
user_injection = True  # Switches injection type
space_charge = True  # Controls field solve on/off
simulateIonization = False  # Include ionization in simulation

if user_injection:
    # User injection thermionic_rz_injector method uses a r**2 scaling to distribute particles uniformly
    variable_weight = False
else:
    # Warp default injection uses radially weighted particles to make injection uniform in r
    variable_weight = True

# Dimensions for the electron cooler
#pipe_radius = 0.1524 / 2. # m (Based on ECE specs)
pipe_radius = 0.03
cooler_length = 30. # m

cathode_temperature = 0.25  # eV

# Beam
beam_beta = 0.990813945176
beam_ke = wp.emass / wp.jperev * wp.clight**2 * (1. / np.sqrt(1-beam_beta**2) - 1.)  # eV
#print '*** beam_ke, beam_gamma =', beam_ke,  1. / np.sqrt(1-beam_beta**2)
beam_current = 3. # A
beam_radius = 0.01  # m

##########################
# 3D Simulation Parameters
##########################

# Set cells (nx == ny in cylindrical coordinates)
wp.w3d.nx = 40
wp.w3d.ny = 40
wp.w3d.nz = 40000

# Set field boundary conditions (Warp only allows setting x and y together)
wp.w3d.bound0 = wp.neumann  # Use neumann to remove field from electrode edge at boundary
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# Set boundary dimensions
wp.w3d.xmmin = -pipe_radius
wp.w3d.xmmax = pipe_radius
wp.w3d.ymmin = -pipe_radius
wp.w3d.ymmax = pipe_radius
wp.w3d.zmmin = 0.0
wp.w3d.zmmax = cooler_length

# Set particle boundary conditions
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb

# Type of particle push. 2 includes tan correction to Boris push.
wp.top.ibpush = 2

dx = (wp.w3d.xmmax - wp.w3d.xmmin) / wp.w3d.nx
dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz  # Because warp doesn't set w3d.dz until after solver instantiated
wp.top.dt = 0.75 * dz / (beam_beta * wp.clight)  # Timestep set to cell crossing time with some padding

####################################
# Create Beam and Set its Parameters
####################################
# parameters for electron beam:
#   total energy: 3.2676813658 MeV
#   beta_e: 0.990813945176

wp.top.lrelativ = True
wp.top.relativity = True

ptcl_per_step = 400 # number of particles to inject on each step
wp.top.npinject = ptcl_per_step

beam = wp.Species(type=wp.Electron, name='Electron', lvariableweights=variable_weight)
if space_charge:
    beam.ibeam = beam_current
else:
    beam.ibeam = 0.0

wp.w3d.l_inj_exact = True  # if true, position and angle of injected particle are
#  computed analytically rather than interpolated

"""
A custom injector routine is used here to allow for a very relativistic beam to be injected directly into the simulation
because Warp's built in routine is not based on relativistic kinematics. Setting user_injection = False should work
well for lower energy beams.
"""

if user_injection:
    wp.top.inject = 6  # Option 6 is a user specified input distribution
    wp.top.ainject = beam_radius  # Width of injection area in x
    wp.top.binject = beam_radius  # Wdith of injection area in y
    injector = UserInjectors(beam, wp.w3d, wp.gchange, cathode_temperature=cathode_temperature,
                             cathode_radius=beam_radius,
                             ptcl_per_step=ptcl_per_step, accelerating_voltage=beam_ke, zmin_scale=0.545)

    wp.installuserinjection(injector.thermionic_rz_injector)
else:
    wp.top.inject = 1  # Option 1 is a constant injection (default: particles injected from xy plane at z=0)
    wp.top.lhalfmaxwellinject = True  # Use half maxwell in axis perpendicular to emission surface (full in other axes)

    beam.a0 = beam_radius  # Width of injected beam in x
    beam.b0 = beam_radius  # Width of injected beam in y
    beam.ap0 = 0.0  # Width of injected beam in vx/vz
    beam.bp0 = 0.0  # Width of injected beam in vy/vz

    beam.vbeam = beam_beta * wp.clight
    assert beam.vbeam < 0.1 * wp.clight, "Injection velocity > 0.1c. " \
                                         "Constant injection does not use relativistic kinematics"
    beam.vthz = np.sqrt(cathode_temperature * wp.jperev / beam.mass)  # z thermal velocity
    beam.vthperp = np.sqrt(cathode_temperature * wp.jperev / beam.mass)  # x and y thermal velocity
    wp.w3d.l_inj_rz = (wp.w3d.solvergeom == wp.w3d.RZgeom)

####################
# Install Conductors
####################
# Create conductors that will represent electrodes placed inside vacuum vessel

pipe_voltage = 0.0  # Set main pipe section to ground
electrode_voltage = +2e2  # electrodes held at several V relative to main pipe
assert electrode_voltage < beam_ke, "Electrodes potential greater than beam KE."

pipe_radius = pipe_radius
electrode_length = 0.25
electrode_gap = 0.1
pipe_length = cooler_length - 2 * electrode_length - 2 * electrode_gap

z_positions = [0.0]
z_positions.append(z_positions[-1] + electrode_length)
z_positions.append(z_positions[-1] + electrode_gap)
z_positions.append(z_positions[-1] + pipe_length)
z_positions.append(z_positions[-1] + electrode_gap)
z_positions.append(z_positions[-1] + electrode_length)

conductors = []

#bottom_cap = wp.Box(xsize=pipe_radius, ysize=pipe_radius, zsize=dz, voltage=electrode_voltage, zcent=.5*dz)
bottom_cap = wp.Box(xsize=pipe_radius, ysize=pipe_radius, zsize=dz, voltage=pipe_voltage, zcent=.5*dz)
#conductors.append(bottom_cap)

left_electrode = wp.Box(xsize=2.*dx, ysize=2.*dx, zsize=dz, voltage=electrode_voltage, xcent=beam_radius+dx, zcent=2.5*dz)
#conductors.append(left_electrode)

entrance_electrode = wp.ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                     zlower=z_positions[0], zupper=z_positions[1])
#conductors.append(entrance_electrode)

beam_pipe = wp.ZCylinderOut(voltage=pipe_voltage, radius=pipe_radius, zlower=0.0, zupper=cooler_length)
#                            zlower=z_positions[2], zupper=z_positions[3])
conductors.append(beam_pipe)

#exit_electrode = wp.ZPlane(z0=cooler_length, zsign=-1., voltage=electrode_voltage)
exit_electrode = wp.ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                 zlower=z_positions[4], zupper=z_positions[5])
#conductors.append(exit_electrode)

right_electrode = wp.Box(xsize=2.*dx, ysize=2.*dx, zsize=dz, voltage=electrode_voltage, xcent=beam_radius+dx, zcent=cooler_length-2.5*dz)
#conductors.append(right_electrode)

#top_cap = wp.Box(xsize=pipe_radius, ysize=pipe_radius, zsize=dz, voltage=electrode_voltage, zcent=cooler_length-.5*dz)
top_cap = wp.Box(xsize=pipe_radius, ysize=pipe_radius, zsize=dz, voltage=pipe_voltage, zcent=cooler_length-.5*dz)
#conductors.append(top_cap)

##############################
# Install Ideal Solenoid Field
##############################
# electron cooler interaction takes place inside a solenoid
# for time being this is represented as an ideal magnetic field B = (0, 0, bz)

# External B-Field can be added by setting vector components at each cell
nxb = wp.w3d.nx // 2
nyb = wp.w3d.ny // 2
nzb = wp.w3d.nz // 2

bx = np.zeros([nxb, nyb, nzb])
by = np.zeros([nxb, nyb, nzb])
bz = np.zeros([nxb, nyb, nzb])

dxb = wp.w3d.xmmax - wp.w3d.xmmin
dyb = wp.w3d.ymmax - wp.w3d.ymmin
dzb = wp.w3d.zmmax - wp.w3d.zmmin

B0 = 0.1 # T
bz[:, :, :] = B0
bz_shape = bz.shape
if wp.comm_world.rank == 0:
    print 'bz_shape =', bz_shape
bi = 5.e-1 * B0
Ncoil = 300
lambda_i = dzb / Ncoil
kzi = 2. * math.pi / lambda_i
phizi = 0.
#kxi = kzi / math.sqrt(2.)
#kyi = kxi
kxi = kzi
kyi = 0.

for j in range(bz_shape[0]):
    xj = wp.w3d.xmmin + j * dxb / nxb
#    if wp.comm_world.rank == 0:
#        print 'xj =', xj
    for k in range(bz_shape[1]):
        yk = wp.w3d.ymmin + k * dyb / nyb
        for l in range(bz_shape[2]):
            zl = wp.w3d.zmmin + l * dzb / nzb
            fxy = math.exp(kxi * xj) * math.exp(kyi * yk)
            fxyz = fxy * math.sin(kzi * zl + phizi)
            bx[j, k, l] = bi * kxi / kzi * fxyz
            by[j, k, l] = bi * kyi / kzi * fxyz
            bz[j, k, l] += bi * fxy * math.cos(kzi * zl + phizi)
#    if wp.comm_world.rank == 0:
#        print 'bx[', j,  bz_shape[1] // 2, bz_shape[2] // 2, '] =', bx[j, bz_shape[1] // 2, bz_shape[2] // 2], bi, kxi / kzi, fxyz

if wp.comm_world.rank == 0:
    np.save('bx', bx)
    np.save('by', by)
    np.save('bz', bz)

# Add B-Field to simulation
wp.addnewbgrd(wp.w3d.zmmin, wp.w3d.zmmax,
              xs = wp.w3d.xmmin, dx = dxb,
              ys = wp.w3d.ymmin, dy = dyb,
              nx = nxb, ny = nyb, nz = nzb,
              bx = bx, by = by, bz = bz)

########################
# Register Field Solvers
########################

# prevent GIST plotting from starting upon setup
wp.top.lprntpara = False
wp.top.lpsplots = False

if space_charge:
    # magnetostatic solver disabled for now, was causing mpi issues and is very slow
    # solverB = MagnetostaticMG()
    # solverB.mgtol = [0.01] * 3
    # registersolver(solverB)

    # Add 2D Field Solve, will be cylindrical solver based on setting of w3d.solvergeom
    solverE = wp.MultiGrid2D()
    wp.registersolver(solverE)

# Conductors must be registered after Field Solver is instantiated if we want them to impact field solve
for cond in conductors:
    wp.installconductor(cond)

# Conductors set as scrapers will remove impacting macroparticles from the simulation
#scraper = wp.ParticleScraper(conductors, lsavecondid=1)
scraper = wp.ParticleScraper(conductors)

######################
# Particle Diagnostics
######################

#wp.top.lsavelostparticles = True

# HDF5 Particle/Field diagnostic options

if particle_diagnostic_switch:
    particleperiod = 1000  # Particle diagnostic write frequency
    particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=wp.top, w3d=wp.w3d,  # Should always be set
                                               # Include data from all existing species in write
                                               species={species.name: species for species in wp.listofallspecies},
                                               # Option for parallel write (if available on system)
                                               comm_world=wp.comm_world, lparallel_output=False,
                                               # `ParticleDiagnostic` automatically append 'hdf5' to path name
                                               write_dir=diagDir[:-5])
    wp.installafterstep(particle_diagnostic_0.write)  # Write method is installed as an after-step action

if field_diagnostic_switch:
    fieldperiod = 1000  # Field diagnostic write frequency
    efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=wp.top, w3d=wp.w3d,
                                                              comm_world=wp.comm_world,
                                                              period=fieldperiod)
    wp.installafterstep(efield_diagnostic_0.write)

    # No B-Field diagnostic since magnetostatic solve turned off

    # bfield_diagnostic_0 = FieldDiagnostic.MagnetostaticFields(solver=solverB, top=top, w3d=w3d,
    #                                                           comm_world=comm_world,
    #                                                           period=fieldperiod)
    # installafterstep(bfield_diagnostic_0.write)

# Crossing Diagnostics
##zcross_l = ZCrossingParticles(zz=z_positions[1], laccumulate=1)
##zcross_r = ZCrossingParticles(zz=z_positions[4], laccumulate=1)
#zcross_l = ZCrossingParticles(zz=5.*dz, laccumulate=1)
#zcross_r = ZCrossingParticles(zz=cooler_length-5.*dz, laccumulate=1)


###########################
# Generate and Run PIC Code
###########################

#electrons_tracked_t0 = wp.Species(type=wp.Electron)
#tracer_count = 50

wp.derivqty()  # Set derived beam properties if any are required
wp.package("w3d")  # Use w3d solver/geometry package
wp.generate()  # Allocate arrays, generate mesh, perform initial field solve

#wp.restart('magnetized_cooler3200000')

Nsteps = 60000
#particle_diagnostic_switch = False
#field_diagnostic_switch = False

while wp.top.it < Nsteps:
    wp.step(1000)
#    if wp.top.it % 100000 == 0 and wp.comm_world.rank == 0:
#        wp.dump()
