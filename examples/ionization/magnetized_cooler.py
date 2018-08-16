from __future__ import division
import warp as wp
import numpy as np

from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.ionization import Ionization
import rswarp.utilities.h2crosssections as h2crosssections
from rswarp.cathode.injectors import UserInjectors
from rswarp.utilities.file_utils import cleanupPrevious
import sys

# Seed set for testing
np.random.seed(123456)

diagDir = 'diags/hdf5'
diagFDir = ['diags/fields/magnetic', 'diags/fields/electric']

# Cleanup command if directories already exist
# Warp's HDF5 diagnostics will not overwrite existing files. This command cleans the diagnostic directory
# to allow for rerunning the simulation
if wp.comm_world.rank == 0:
    cleanupPrevious(diagDir, diagFDir)

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
simulateIonization = True  # Include ionization in simulation

if user_injection:
    # User injection thermionic_rz_injector method uses a r**2 scaling to distribute particles uniformly
    variable_weight = False
else:
    # Warp default injection uses radially weighted particles to make injection uniform in r
    variable_weight = True

# Dimensions for the electron cooler
pipe_radius = 0.1524 / 2.  # m (Based on ECE specs)
cooler_length = 2.0  # m

cathode_temperature = 0.25  # eV

# Beam
beam_beta = 0.990813945176
beam_ke = wp.emass / wp.jperev * wp.clight**2 * (1. / np.sqrt(1-beam_beta**2) - 1.)  # eV
print '*** beam_ke, beam_gamma =', beam_ke,  1. / np.sqrt(1-beam_beta**2)
#sys.exit(0)
beam_current = 10e-3  # A
beam_radius = 0.01  # m

##########################
# 3D Simulation Parameters
##########################

# Set cells (nx == ny in cylindrical coordinates)
wp.w3d.nx = 128
wp.w3d.ny = 128
wp.w3d.nz = 1024

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

dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz  # Because warp doesn't set w3d.dz until after solver instantiated
wp.top.dt = 0.5 * dz / (beam_beta * wp.clight)  # Timestep set to cell crossing time with some padding

####################################
# Create Beam and Set its Parameters
####################################
# parameters for electron beam:
#   total energy: 3.2676813658 MeV
#   beta_e: 0.990813945176

wp.top.lrelativ = True
wp.top.relativity = True

ptcl_per_step = 1000  # number of particles to inject on each step
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

##############################
# Ionization of background gas
##############################

# Ionization e- + H2 -> H2+ + e- + e-

if simulateIonization is True:
    # Particle species for emission products of an ionization event
    h2plus = wp.Species(type=wp.Dihydrogen, charge_state=+1, name='H2+', weight=2)
    emittedelec = wp.Species(type=wp.Electron, name='emitted e-', weight=2)

    target_pressure = 0.4  # in Pa
    target_temp = 273  # in K
    target_density = target_pressure / wp.boltzmann / target_temp  # in 1/m^3

    # Instantiate Ionization
    # Dimensions control where background gas reservoir will be present in the domain
    ioniz = Ionization(
        stride=100,  # Number of particles allowed to be involved in ionization events per timestep
        xmin=wp.w3d.xmmin,
        xmax=wp.w3d.xmmax,
        ymin=wp.w3d.ymmin,
        ymax=wp.w3d.ymmax,
        zmin=wp.w3d.zmmin,
        zmax=wp.w3d.zmmax,
        nx=wp.w3d.nx,
        ny=wp.w3d.ny,
        nz=wp.w3d.nz,
        l_verbose=True
    )

    # add method used to add possible ionization events

    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, emittedelec],  # iterable of species created from ionization
        cross_section=h2crosssections.h2_ioniz_crosssection,  # Cross section, can be float or function
        emitted_energy0=['thermal', h2crosssections.ejectedEnergy],  # Energy of each emitted species, can be float or function
        # or set to 'thermal' to create ions with a thermal energy spread set by temperature
        emitted_energy_sigma=[0, 0],  # Energy spread of emitted species (gives width of gaussian distribution)
        temperature=[target_temp, None],
        sampleEmittedAngle=h2crosssections.generateAngle,
        writeAngleDataDir=False,  # Write file recording statistics of angles
        writeAnglePeriod=1000,  # Period to write angle data, if used
        l_remove_incident=False,  # Remove incident macroparticles involved in ionization
        l_remove_target=False,  # Remove target macroparticles (only used if target_species set)
        ndens=target_density  # Background gas density (if target is reservoir of gas and not macroparticles)
    )

####################
# Install Conductors
####################
# Create conductors that will represent electrodes placed inside vacuum vessel

pipe_voltage = 0.0  # Set main pipe section to ground

# Electrodes turned off for thermal ion distribution test
electrode_voltage = +0.0  # electrodes held at +2.3 relative to main pipe

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

entrance_electrode = wp.ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                     zlower=z_positions[0], zupper=z_positions[1])
conductors.append(entrance_electrode)

beam_pipe = wp.ZCylinderOut(voltage=pipe_voltage, radius=pipe_radius,
                            zlower=z_positions[2], zupper=z_positions[3])
conductors.append(beam_pipe)

exit_electrode = wp.ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                 zlower=z_positions[4], zupper=z_positions[5])
conductors.append(exit_electrode)

##############################
# Install Ideal Solenoid Field
##############################
# electron cooler interaction takes place inside a solenoid
# for time being this is represented as an ideal magnetic field B = (0, 0, bz)

# External B-Field can be added by setting vector components at each cell
bz = np.zeros([wp.w3d.nx, wp.w3d.ny, wp.w3d.nz])
bz[:, :, :] = 1.0  # T
z_start = wp.w3d.zmmin
z_stop = wp.w3d.zmmax

# Add B-Field to simulation
wp.addnewbgrd(z_start, z_stop,
              xs=wp.w3d.xmmin, dx=(wp.w3d.xmmax - wp.w3d.xmmin),
              ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin),
              nx=wp.w3d.nx, ny=wp.w3d.ny, nz=wp.w3d.nz, bz=bz)

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
scraper = wp.ParticleScraper(conductors)

######################
# Particle Diagnostics
######################

# HDF5 Particle/Field diagnostic options

if particle_diagnostic_switch:
    particleperiod = 50  # Particle diagnostic write frequency
    particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=wp.top, w3d=wp.w3d,  # Should always be set
                                               # Include data from all existing species in write
                                               species={species.name: species for species in wp.listofallspecies},
                                               # Option for parallel write (if available on system)
                                               comm_world=wp.comm_world, lparallel_output=False,
                                               # `ParticleDiagnostic` automatically append 'hdf5' to path name
                                               write_dir=diagDir[:-5])
    wp.installafterstep(particle_diagnostic_0.write)  # Write method is installed as an after-step action

if field_diagnostic_switch:
    fieldperiod = 100  # Field diagnostic write frequency
    efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=wp.top, w3d=wp.w3d,
                                                              comm_world=wp.comm_world,
                                                              period=fieldperiod)
    wp.installafterstep(efield_diagnostic_0.write)

    # No B-Field diagnostic since magnetostatic solve turned off

    # bfield_diagnostic_0 = FieldDiagnostic.MagnetostaticFields(solver=solverB, top=top, w3d=w3d,
    #                                                           comm_world=comm_world,
    #                                                           period=fieldperiod)
    # installafterstep(bfield_diagnostic_0.write)


###########################
# Generate and Run PIC Code
###########################

wp.derivqty()  # Set derived beam properties if any are required
wp.package("w3d")  # Use w3d solver/geometry package
wp.generate()  # Allocate arrays, generate mesh, perform initial field solve

# Run PIC loop for x steps
wp.step(3000)
