from __future__ import division
import numpy as np
from warp import *
from rswarp.utilities.ionization import Ionization
from rswarp.utilities.beam_distributions import createKV
import shutil
from shutil import os

simulateIonization = True

beam_ke = 100  # beam kinetic energy, in eV
beam_gamma = beam_ke/511e3 + 1
beam_beta = np.sqrt(1-1/beam_gamma**2)
sw = 1

# Weights are set to zero here to disable field solve in the interest of speed
beam = Species(type=Electron, name='e-', weight=0)
# These two species represent the emitted particles
h2plus = Species(type=Dihydrogen, charge_state=+1, name='H2+', weight=0)
emittedelec = Species(type=Electron, name='emitted e-', weight=0)

beam.ibeam = 1e-6

top.dt = 0.1e-9
ptcl_per_step = int(beam.ibeam * top.dt / echarge / sw)  # number of particles to inject on each step


def cleanupPrevious(outputDirectory='angleDiagnostic'):
    if os.path.exists(outputDirectory):
        shutil.rmtree(outputDirectory, ignore_errors=True)

cleanupPrevious()

################################
# 3D Simulation Parameters     #
################################

# Set cells
w3d.nx = 32
w3d.ny = 32
w3d.nz = 64

# Set boundaries
w3d.xmmin = -0.16
w3d.xmmax = 0.16
w3d.ymmin = -0.16
w3d.ymmax = 0.16
w3d.zmmin = 0.00
w3d.zmmax = 0.20

top.pbound0 = absorb
top.pboundnz = absorb
top.pboundxy = absorb

Lz = (w3d.zmmax - w3d.zmmin)
dz =  Lz / w3d.nz
top.dt = (dz) / (beam_beta * clight) / 3  # 3 timesteps to cross a single cell

top.ibpush = 1  # 0:off, 1:fast, 2:accurate

top.lrelativ = True
top.relativity = 1

# --- Other injection variables
w3d.l_inj_exact = True
w3d.l_inj_area = False

w3d.solvergeom = w3d.RZgeom

w3d.bound0 = periodic
w3d.boundnz = periodic
w3d.boundxy = dirichlet

package("w3d")
generate()


def generateDist(npart=ptcl_per_step, zemit=dz/5, dv_over_v=0):
    ptclTrans = createKV(
        npart=npart,
        a=0.010,
        b=0.010,
        emitx=4. * 1.e-6,
        emity=4. * 1.e-6
    )

    if dv_over_v > 0:
        vzoffset = np.random.normal(scale=dv_over_v, size=npart)
    else:
        vzoffset = np.reshape(np.zeros(npart), (npart, 1))

    vz = beam_beta * clight * (np.ones(npart) + vzoffset)
    zemit = zemit * np.ones(npart)
    return np.column_stack((ptclTrans, zemit, vz))


def injectelectrons(npart=ptcl_per_step, zoffset=w3d.dz/5):  # emitting surface 1/5th of a cell forward by default
    ptclArray = generateDist(npart=npart, zemit=zoffset, dv_over_v=0.003)
    beam.addparticles(
        x=ptclArray[:, 0],
        y=ptclArray[:, 2],
        z=ptclArray[:, 4],
        vx=ptclArray[:, 1] * ptclArray[:, 5],  # Change to x and y angles to velocities
        vy=ptclArray[:, 3] * ptclArray[:, 5],
        vz=ptclArray[:, 5]
    )

installuserinjection(injectelectrons)

# --- Specify injection of the particles
top.inject = 6                       # 2 means space-charge limited injection, 6 is user specified
top.npinject = ptcl_per_step           # Approximate number of particles injected each step
top.ainject = 0.0008                      # Must be set even for user defined injection, doesn't seem to do anything
top.binject = 0.0008                      # Must be set even for user defined injection, doesn't seem to do anything

####################################
# Ionization of background gas     #
####################################

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
        zmin=(w3d.zmmin + w3d.zmmax)/2. - w3d.dz*3,
        zmax=(w3d.zmmin + w3d.zmmax)/2. + w3d.dz*3,
        nx=w3d.nx,
        ny=w3d.ny,
        nz=w3d.nz,
        l_verbose=True
    )

    # e + H2 -> 2e + H2+
    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, emittedelec],
        cross_section=1e-25,
        emitted_energy0=[0, lambda nnew, vi: 1./np.sqrt(1-((vi/2.)/clight)**2) * emass * clight**2],
        emitted_energy_sigma=[0, lambda nnew, vi: 0],
        sampleEmittedAngle=lambda nnew, emitE, incE: np.random.uniform(0, 2*np.pi),
        sampleIncidentAngle=lambda nnew, emitE, incE, emitTheta: np.random.uniform(0, 2*np.pi),
        writeAngleDataDir='angleDiagnostic',
        writeAnglePeriod=100,
        l_remove_incident=False,
        l_remove_target=False,
        ndens=target_density
    )

derivqty()  # Sets addition derived parameters (such as beam.vbeam)

##########################
# Injection Controls     #
##########################

# --- Specify injection of the particles
top.inject = 6                       # 2 means space-charge limited injection, 6 is user specified
top.npinject = ptcl_per_step           # Approximate number of particles injected each step
top.ainject = 0.0008                      # Must be set even for user defined injection, doesn't seem to do anything
top.binject = 0.0008                      # Must be set even for user defined injection, doesn't seem to do anything

package("w3d")
generate()

stept(100e-9)  # Simulate 100 ns
