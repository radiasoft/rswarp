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
import pickle
from datetime import datetime

sys.path.append('/Users/chall/research/github/rswarp/')

from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.ionization import Ionization
from rswarp.utilities.beam_distributions import createKV
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

# Switches
particle_diagnostic_switch = False
field_diagnostic_switch = False

pipe_radius = 0.1524 / 2.  # Based on ECE specs
cooler_length = 2.0

####################################
# Create Beam and Set its Parameters
####################################

SC = True  # Controls field solve
simulateIonization = True

ptcl_per_step = 80000  # number of particles to inject on each step

beam_beta = 0.56823  # v/c for 110 keV electrons
beam_current = 10e-3

beam_weight = 0.5 * beam_current / (echarge * beam_beta * clight) / ptcl_per_step

top.lrelativ = True
top.relativity = 1

beam = Species(type=Electron, name='Electron', weight=beam_weight)
# These two species represent the emitted particles
h2plus = Species(type=Dihydrogen, charge_state=+1, name='H2+', weight=2)
emittedelec = Species(type=Electron, name='emitted e-', weight=1)

if SC == False:
    beam.sw = 0.0  # Turn off SC


def generateDist():
    ptclTrans = createKV(
        npart=ptcl_per_step,
        a=0.010,
        b=0.010,
        emitx=4. * 1.e-6,
        emity=4. * 1.e-6
    )

    zrand = np.random.rand(ptcl_per_step, )
    zvel = np.ones_like(zrand) * beam_beta * clight

    return np.column_stack((ptclTrans, zrand, zvel))


def createmybeam():
    ptclArray = generateDist()
    beam.addparticles(x=ptclArray[:, 0], y=ptclArray[:, 2], z=ptclArray[:, 4],
                      vx=ptclArray[:, 1] * ptclArray[:, 5], vy=ptclArray[:, 3] * ptclArray[:, 5], vz=ptclArray[:, 5])


################################
### 3D Simulation Parameters ###
################################

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
top.pbound0 = periodic
top.pboundnz = periodic
# top.pboundxy = absorb

top.ibpush = 2

dz = (w3d.zmmax - w3d.zmmin) / w3d.nx  # Because warp doesn't set w3d.dz until after solver instantiated
top.dt = dz / (beam_beta * 3e8)

##################
# Ionization Setup
##################

##############################
# Ionization of background gas
##############################

if simulateIonization is True:
    target_pressure = 1  # in Pa
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
# Injection Controls
####################

top.inject = 6  # 6 is user specified
top.npinject = ptcl_per_step * top.dt * clight * beam_beta / (w3d.zmmax - w3d.zmmin)  # Approximate number of particles injected each step
# or average number of particles in interval of a step
# will determine current if ibeam is set and beam.sw = 0

top.ainject = 0.0008  # Must be set even for user defined injection, doesn't seem to do anything
top.binject = 0.0008  # Must be set even for user defined injection, doesn't seem to do anything

w3d.l_inj_exact = True  # if true, position and angle of injected particle are
# computed analytically rather than interpolated
w3d.l_inj_area = False  # Not sure what this does


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

w3d.solvergeom = w3d.RZgeom

# prevent GIST from starting upon setup
top.lprntpara = False
top.lpsplots = False

package("w3d")  # package/generate Must be called after geometry is set
generate()

if SC == True:
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
    particleperiod = 1000
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
    # installafterstep(efield_diagnostic_0.write)
    # bfield_diagnostic_0 = FieldDiagnostic.MagnetostaticFields(solver=solverB, top=top, w3d=w3d,
    #                                                           comm_world=comm_world,
    #                                                           period=fieldperiod)
    # installafterstep(bfield_diagnostic_0.write)


###########################
# Generate and Run PIC Code
###########################

installparticleloader(
    createmybeam)  # for particleloader the call Must be between 1st and 2nd generate calls (or macroparticles double)

package("w3d")  # package/generate must be called a second time, after solver set
generate()

step(1000)


