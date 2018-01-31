"""
Note: Due to particle loading process package/generate commands must be given twice and carefully sequenced
1. Set geometry
2. package/generate
3. Field solvers instantiated
4. Particle loading
5. package/generate
"""

from __future__ import division
from warp import *
import numpy as np
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
import sys

sys.path.append('/Users/chall/research/github/rswarp/')

from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.ionization import Ionization
from rswarp.utilities.file_utils import cleanupPrevious
import rsoopic.h2crosssections as h2crosssections
diagDir = 'diags/hdf5'
diagFDir = ['diags/fields/magnetic', 'diags/fields/electric']

# Cleanup command if directories already exist
if comm_world.rank == 0:
    cleanupPrevious(diagDir, diagFDir)

####################
# General Parameters
####################
w3d.solvergeom = w3d.RZgeom

# Switches
particle_diagnostic_switch = False
field_diagnostic_switch = False

pipe_radius = 0.1524 / 2.  # Based on ECE specs
cooler_length = 2.0

cathode_temperature = 0.25
####################################
# Create Beam and Set its Parameters
####################################
# parameters for electron beam:
#   total energy: 3.2676813658 MeV
#   beta_e: 0.990813945176

space_charge = True  # Controls field solve

top.inject = 1  # Constant injection
top.lrelativ = True
top.relativity = True  # TODO: Figure out what this actually does and how it should be set
top.lhalfmaxwellinject = True

ptcl_per_step = 1000  # number of particles to inject on each step
top.npinject = ptcl_per_step

beam_beta = 0.990813945176
beam_current = 10e-3
beam_radius = 0.01

# if space_charge:
#     beam_weight = 0.5 * beam_current / (echarge * beam_beta * clight) / ptcl_per_step
# else:
#     beam_weight = 0.0

beam = Species(type=Electron, name='Electron')

beam.a0 = beam_radius
beam.b0 = beam_radius
beam.ap0 = 0.0
beam.bp0 = 0.0

beam.vbeam = beam_beta * clight
beam.vthz = sqrt(cathode_temperature * jperev / beam.mass)
beam.vthperp = sqrt(cathode_temperature * jperev / beam.mass)

if space_charge:
    beam.ibeam = beam_current
else:
    beam.ibeam = 0.0

w3d.l_inj_exact = True  # if true, position and angle of injected particle are
# computed analytically rather than interpolated
w3d.l_inj_rz = (w3d.solvergeom == w3d.RZgeom)

##########################
# 3D Simulation Parameters
##########################

# Set cells
w3d.nx = 128
w3d.ny = 128
w3d.nz = 1024

w3d.bound0 = neumann  # Use neumann to remove field from electrode edge at boundary
w3d.boundnz = neumann
w3d.boundxy = neumann

# Set boundaries
w3d.xmmin = -pipe_radius
w3d.xmmax = pipe_radius
w3d.ymmin = -pipe_radius
w3d.ymmax = pipe_radius
w3d.zmmin = 0.0
w3d.zmmax = cooler_length

# Longitudinal absorbing boundaries off to allow beam to recirculate
top.pbound0 = absorb
top.pboundnz = absorb
# top.pboundxy = absorb

top.ibpush = 2

dz = (w3d.zmmax - w3d.zmmin) / w3d.nx  # Because warp doesn't set w3d.dz until after solver instantiated
top.dt = dz / (beam_beta * 3e8)

##################
# Ionization Setup
##################
simulateIonization = True

# These two species represent the emitted particles
h2plus = Species(type=Dihydrogen, charge_state=+1, name='H2+', weight=2)
emittedelec = Species(type=Electron, name='emitted e-', weight=1)

##############################
# Ionization of background gas
##############################

# TODO: Check gas pressure

if simulateIonization is True:
    target_pressure = 1.0  # in Pa
    target_temp = 273  # in K
    target_density = target_pressure / boltzmann / target_temp  # in 1/m^3

    ioniz = Ionization(
        stride=100,
        xmin=w3d.xmmin,
        xmax=w3d.xmmax,
        ymin=w3d.ymmin,
        ymax=w3d.ymmax,
        zmin=w3d.zmmin,
        zmax=w3d.zmmax,
        nx=w3d.nx,
        ny=w3d.ny,
        nz=w3d.nz,
        l_verbose=True
    )

    # # e + H2 -> 2e + H2+
    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, emittedelec],
        cross_section=h2crosssections.h2_ioniz_crosssection,
        emitted_energy0=[0, h2crosssections.ejectedEnergy],
        emitted_energy_sigma=[0, 0],
        sampleEmittedAngle=h2crosssections.generateAngle,
        writeAngleDataDir=False,  # diagDir + '/angleDiagnostic',
        writeAnglePeriod=1000,
        l_remove_incident=False,
        l_remove_target=False,
        ndens=target_density
    )

derivqty()

####################
# Install Conductors
####################

pipe_voltage = 0.0  # Set main pipe section to ground
electrode_voltage = -2e3  # electrodes held at several kV relative to main pipe

pipe_radius = pipe_radius  # Reminder
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

entrance_electrode = ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                  zlower=z_positions[0], zupper=z_positions[1])
conductors.append(entrance_electrode)

beam_pipe = ZCylinderOut(voltage=pipe_voltage, radius=pipe_radius,
                         zlower=z_positions[2], zupper=z_positions[3])
conductors.append(beam_pipe)

exit_electrode = ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                              zlower=z_positions[4], zupper=z_positions[5])
conductors.append(exit_electrode)

##############################
# Install Ideal Solenoid Field
##############################

# Uniform B_z field
bz = np.zeros([w3d.nx, w3d.ny, w3d.nz])
bz[:, :, :] = 1.0
z_start = w3d.zmmin
z_stop = w3d.zmmax

addnewbgrd(z_start, z_stop, xs=w3d.xmmin, dx=(w3d.xmmax - w3d.xmmin), ys=w3d.ymmin, dy=(w3d.ymmax - w3d.ymmin),
           nx=w3d.nx, ny=w3d.ny, nz=w3d.nz, bz=bz)

########################
# Register Field Solvers
########################

# prevent GIST from starting upon setup
top.lprntpara = False
top.lpsplots = False

if space_charge:
    solverB = MagnetostaticMG()
    solverB.mgtol = [0.01] * 3
    registersolver(solverB)
    solverE = MultiGrid2D()
    registersolver(solverE)

# Register conductors and set as particle scrapers
for cond in conductors:
    installconductor(cond)
scraper = ParticleScraper(conductors)

######################
# Particle Diagnostics
######################

# Particle/Field diagnostic options
if particle_diagnostic_switch:
    particleperiod = 100
    particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=top, w3d=w3d,
                                               species={species.name: species for species in listofallspecies},
                                               comm_world=comm_world, lparallel_output=False,
                                               write_dir=diagDir[:-5])
    installafterstep(particle_diagnostic_0.write)

if field_diagnostic_switch:
    fieldperiod = 1000
    efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d,
                                                              comm_world=comm_world,
                                                              period=fieldperiod)
    installafterstep(efield_diagnostic_0.write)
    bfield_diagnostic_0 = FieldDiagnostic.MagnetostaticFields(solver=solverB, top=top, w3d=w3d,
                                                              comm_world=comm_world,
                                                              period=fieldperiod)
    installafterstep(bfield_diagnostic_0.write)


###########################
# Generate and Run PIC Code
###########################

package("w3d")  # package/generate must be called a second time, after solver set
generate()

step(1000)


