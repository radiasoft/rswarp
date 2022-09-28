from __future__ import division

from warp import *
from warp.data_dumping.openpmd_diag.particle_diag import ParticleDiagnostic

from rswarp.ionization.ionization import Ionization
from rswarp.utilities.beam_distributions import createKV
from rswarp.utilities.file_utils import cleanupPrevious

import rsoopic.h2crosssections as h2crosssections
import rswarp.ionization.crosssections as Xsect

from shutil import os

# prevent gist from starting upon setup
top.lprntpara = false
top.lpsplots = false

simulateIonization = True

beam_ke = 1e3  # beam kinetic energy, in eV
beam_gamma = beam_ke/511e3 + 1
beam_beta = np.sqrt(1-1/beam_gamma**2)
sw = 1

diagDir = ("diags-%.3fkeV" % (beam_ke/1e3))

beam = Species(type=Electron, name='e-', weight=sw)
# These two species represent the emitted particles
h2plus = Species(type=Dihydrogen, charge_state=+1, name='H2+', weight=sw)
emittedelec = Species(type=Electron, name='emitted e-', weight=sw)

beam.ibeam = 1e-6

top.lrelativ = True
top.relativity = 1

# Directory paths
field_base_path = os.path.join(diagDir, 'fields')
diagFDir = {'magnetic': os.path.join(field_base_path, 'magnetic'),
            'electric': os.path.join(field_base_path, 'electric')}

# Cleanup command if directories already exist
if comm_world.rank == 0:
    cleanupPrevious(diagDir, diagFDir)

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
top.dt = 0.1e-9
#import sys; from scipy.constants import c; print beam_beta, beam_beta * c * top.dt, '<', dz, '?'; sys.exit(0)
ptcl_per_step = int(beam.ibeam * top.dt / echarge / sw)  # number of particles to inject on each step

top.ibpush = 1  # 0:off, 1:fast, 2:accurate

# --- Other injection variables
w3d.l_inj_exact = True
w3d.l_inj_area = False

w3d.solvergeom = w3d.RZgeom

w3d.bound0 = dirichlet
w3d.boundnz = dirichlet
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
        zmin=w3d.zmmin,
        zmax=w3d.zmmax,
        nx=w3d.nx,
        ny=w3d.ny,
        nz=w3d.nz,
        l_verbose=True
    )

    # # e + H2 -> 2e + H2+
    h2xs = Xsect.H2IonizationEvent()
    def xswrapper(vi):
        return h2xs.getCrossSection(vi)
    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, emittedelec],
        #cross_section=h2crosssections.h2_ioniz_crosssection,
        # cross_section=lambda nnew, vi: 1e-20,
        cross_section=xswrapper,
        emitted_energy0=[0, h2crosssections.ejectedEnergy],
        # emitted_energy0=[0, lambda nnew, vi: 1./np.sqrt(1.-((vi/2.)/clight)**2) * emass*clight/jperev],
        emitted_energy_sigma=[0, 0],
        # sampleEmittedAngle=lambda nnew, emitted_energy, incident_energy: np.random.uniform(0, 2*np.pi, size=nnew),
        sampleEmittedAngle=h2crosssections.generateAngle,
        # sampleIncidentAngle=lambda nnew, emitted_energy, incident_energy, emitted_theta: np.random.uniform(0, 2*np.pi, size=nnew),
        writeAngleDataDir=diagDir,
        writeAnglePeriod=1,
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

diagP = ParticleDiagnostic(
    period=50,
    top=top,
    w3d=w3d,
    species={species.name: species for species in listofallspecies},
    comm_world=comm_world,
    lparallel_output=False,
    write_dir=diagDir
)

diags = [diagP]

def writeDiagnostics():
    for d in diags:
        d.write()

# installafterstep(writeDiagnostics)

package("w3d")
generate()

# stept(100e-9)  # Simulate 100 ns
random.seed(1234)
step(50)

print(h2plus.nps, emittedelec.nps)
# expected: 1020 1001


