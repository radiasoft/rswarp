from __future__ import division
import pytest
import numpy as np

from warp import *
from warp.data_dumping.openpmd_diag.particle_diag import ParticleDiagnostic

from rswarp.utilities.ionization import Ionization
from rswarp.utilities.beam_distributions import createKV
from rswarp.utilities.file_utils import cleanupPrevious

import rsoopic.h2crosssections as h2crosssections
# Load module with new, relativistic cross section
sys.path.insert(1, '/home/vagrant/jupyter/rswarp/rswarp/ionization')
import crosssections as Xsect
h2xs = Xsect.H2IonizationEvent()

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt

#import sys

beam_radius = 1.e-2

def test_ionization():

    # prevent gist from starting upon setup
    top.lprntpara = false
    top.lpsplots = false

    simulateIonization = True

    beam_ke = 1.0e3 # beam kinetic energy, in eV
    beam_gamma = beam_ke/511.e3 + 1
    beam_beta = np.sqrt(1-1/beam_gamma**2)
    #print 'beam_beta =', beam_beta; sys.exit(0)
    sw = 1

    beam = Species(type=Electron, name='e-', weight=sw)
    # These two species represent the emitted particles
    h2plus = Species(type=Dihydrogen, charge_state=+1, name='H2+', weight=sw)
    emittedelec = Species(type=Electron, name='emitted e-', weight=sw)

    beam.ibeam = 1e-6

    top.lrelativ = True
    top.relativity = 1

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
    # number of particles to inject on each step
    ptcl_per_step = int(beam.ibeam * top.dt / echarge / sw)

    top.ibpush = 1 # 0:off, 1:fast, 2:accurate

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
            a=beam_radius,
            b=beam_radius,
            emitx=4.* 1.e-6,
            emity=4.* 1.e-6
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
        def xswrapper(vi):
            sigarr = np.empty((vi.size))
            for i in range(vi.size):
                sigarr[i] = h2xs.getCrossSection(vi[i])
            return sigarr
        ioniz.add(
            incident_species=beam,
            emitted_species=[h2plus, emittedelec],
            #cross_section=lambda nnew, vi: 1e-20,
            #cross_section=h2crosssections.h2_ioniz_crosssection,
            #cross_section=Xsect.IonizationTarget.getCrossSection,
            cross_section=xswrapper,
            emitted_energy0=[0, h2crosssections.ejectedEnergy],
            # emitted_energy0=[0, lambda nnew, vi: 1./np.sqrt(1.-((vi/2.)/clight)**2) * emass*clight/jperev],
            emitted_energy_sigma=[0, 0],
            # sampleEmittedAngle=lambda nnew, emitted_energy, incident_energy: np.random.uniform(0, 2*np.pi, size=nnew),
            sampleEmittedAngle=h2crosssections.generateAngle,
            # sampleIncidentAngle=lambda nnew, emitted_energy, incident_energy, emitted_theta: np.random.uniform(0, 2*np.pi, size=nnew),
            writeAngleDataDir=False,
            writeAnglePeriod=10,
            l_remove_incident=False,
            l_remove_target=False,
            ndens=target_density
        )

    derivqty() # Sets addition derived parameters (such as beam.vbeam)

##########################
# Injection Controls     #
##########################

    # --- Specify injection of the particles
    top.inject = 6               # 2 means space-charge limited injection, 6 is user specified
    top.npinject = ptcl_per_step # Approximate number of particles injected each step
    top.ainject = 0.0008         # Must be set even for user defined injection, doesn't seem to do anything
    top.binject = 0.0008         # Must be set even for user defined injection, doesn't seem to do anything

    package("w3d")
    generate()

    Tsim = 10.0e-9 # Simulate 10 ns
    stept(Tsim)

    # Compare emitted electrons created in simulation (Nes) with analytic estimate (Ne)
    v_b = clight * beam_beta # beam speed
    #sigma = h2crosssections.h2_ioniz_crosssection(v_b)
    # Calculate cross section using Kim RBEB model
    sigma = h2xs.getCrossSection(v_b)
    A_b = math.pi * beam_radius**2 # beam cross section
    n_e = beam.ibeam / (jperev * A_b * v_b) # electron density inside beam
    Ne = sigma * target_density * n_e * v_b * A_b
    if Tsim > Lz / v_b: # beam reaches far side of simulation domain
        Ne *= v_b * (Lz / v_b)**2 / 2. + Lz * (Tsim - Lz / v_b)
    else:
        Ne *= v_b * Tsim**2 / 2.
    Ne = 0.5 * math.pi * sigma * target_density * n_e
    Ne *= (beam_radius * v_b * Tsim)**2
    Nes = emittedelec.getn()
    eps_Ne = (Nes - Ne) / Nes
    if __name__ == '__main__':
        print ' ***', Nes, Ne, eps_Ne
    assert(eps_Ne < 4.e-2) # tolerate relative error less than 4%

    #plt.plot(emittedelec.getx(), emittedelec.gety(), '.')
    #plt.savefig('beamXsection.png')

    # Check that number of emitted electrons is within expected range
    if __name__ == '__main__':
        print emittedelec.getn() # should be around 3650
    np.testing.assert_approx_equal(emittedelec.getn()-150, 3500, 1) # accept 3150 <= n < 4150
    # Check that mean and standard deviation is reasonable for all six phasespace coordinates
    if __name__ == '__main__':
        print np.mean(emittedelec.getx()), np.std(emittedelec.getx())
    np.testing.assert_approx_equal(np.std(emittedelec.getx()), 9.0e-3, 1)
    if __name__ == '__main__':
        print np.mean(emittedelec.gety()), np.std(emittedelec.gety())
    np.testing.assert_approx_equal(np.std(emittedelec.gety()), 9.0e-3, 1)
    if __name__ == '__main__':
        print np.mean(emittedelec.getz()), np.std(emittedelec.getz())
    np.testing.assert_approx_equal(np.std(emittedelec.getz()), 4.0e-2, 1)
    if __name__ == '__main__':
        print np.mean(emittedelec.getvx()), np.std(emittedelec.getvx())
    np.testing.assert_approx_equal(np.std(emittedelec.getvx()), 1.9e6, 2)
    if __name__ == '__main__':
        print np.mean(emittedelec.getvy()), np.std(emittedelec.getvy())
    np.testing.assert_approx_equal(np.std(emittedelec.getvy()), 1.9e6, 2)
    if __name__ == '__main__':
        print np.mean(emittedelec.getvz()), np.std(emittedelec.getvz())
    np.testing.assert_approx_equal(np.mean(emittedelec.getvz()), 5.0e5, 1)
    np.testing.assert_approx_equal(np.std(emittedelec.getvz()), 1.5e6, 2)

if __name__ == '__main__':
    test_ionization()
