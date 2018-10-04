#!/usr/bin/env python
"""
Utility module for calculating ionization cross sections, a la:

[1] Yong-Ki Kim, Jose Paulo Santos, and Fernando Parente,
    "Extension of the binary-encounter-dipole model to relativistic
    incident electrons", Phys. Rev. A 62, 052710, 13 October 2000
    <http://teddy.physics.utah.edu/papers/physrev/PRA52710.pdf>
[2] Nikolai G. Lehtinen, "Relativistic runaway electrons above thunderstorms",
    Stanford University PhD Thesis, 2000
    <http://nova.stanford.edu/~vlf/publications/theses/nlehtthesis.pdf>
[3] D. Bruhwiler, "RNG Calculations for Scattering in XOOPIC", Tech-X Note, 2000
[4] M. E. Rudd, Y.-K. Kim, D. H. Madison and J. W. Gallagher,
    "Electron production in proton collisions: total cross sections",
    Reviews of Modern Physics, Vol. 57, No. 4, October 1985
    <https://digitalcommons.unl.edu/cgi/viewcontent.cgi?article=1041&context=physicsrudd>
[5] Ya. B. Zeldovich and Yu. P. Rayzer,
    "Physics of shock waves and high-temperature hydrodynamic phenomena",
    FTD-MT 64-514, 30 September 1965
    <http://www.dtic.mil/dtic/tr/fulltext/u2/622356.pdf>
"""
from __future__ import division
import math
import random
#import time
import numpy as np
# any import from warp will trigger arg parsing, which will fail without this
# in a notebook context (or anything else with non-warp commandline args)
import warpoptions
warpoptions.ignoreUnknownArgs = True
from warp import emass, clight, jperev
from scipy.constants import physical_constants, fine_structure # fine-structure constant
import rsoopic.h2crosssections as h2crosssections

echarge = jperev
pmass = 1.8362e3 * emass
H2mass = 1.66e-27 # mass of hydrogen molecule (in kg)

class IonizationEvent:
    """
    Base class for all ionization events. Uses the Moller cross section
    for electron-impact ionization (Eq. 2.15 in Ref. [2]), which has a
    free parameter eps_min.
    The getCrossSection method is intended to be customized for specific
    target atom/molecule/ion.
    """
    R = 13.60569 # Rydberg energy (in eV)
    a_0 = physical_constants['Bohr radius'][0]

    emassEV = emass * clight**2 / jperev

    useMollerApproximation = True

    # Value of eps_min for which Kim [1] and Lehtinen [2] versions
    # of Moller agree for H2
    eps_min = 8. # (in eV)

    class Target: # container for target-particle parameters
        def __init__(self, number_of_electrons, mass, charge):
            self.N = number_of_electrons
            if mass is not None: self.mass = mass
            if charge != 0.: self.charge = charge

    class Projectile: # container for projectile-particle parameters
        def __init__(self, mass, charge):
            self.mass = mass
            self.charge = charge

    def __init__(self, trgN = 1, trgm = None, trgq = 0.):
        self.target = Target(trgN, trgm, trgq)
        self.projectile = Projectile(emass, echarge)

    def setEps_min(self, val):
        self.eps_min = val

    def getCrossSection(self, vi):
        """
        Compute and return the total cross-section for electron impact ionization
        (in units of m**2)
        vi - incident electron velocity (in units of m/s, this is passed in from
        warp as vxi = uxi * gaminvi)
        Cross section sigma is given by Eq. 2.15 in Ref. [2]
        """
        # initialize needed variables
        r_0 = physical_constants['classical electron radius'][0]
        mec2 = emass * clight**2
        beta_t = vi / clight
        kappa = 2. * math.pi * r_0**2 * mec2 / (beta_t * beta_t)
        eps_min_joule = self.eps_min * jperev
        eps = mec2 * (1. / math.sqrt(1. - beta_t**2) - 1.) # incident energy (in J)
        # now add terms to sigma, one by one:
        sigma = 1. / eps_min_joule - 1. / (eps - eps_min_joule)
        sigma += (eps - 2. * eps_min_joule) / (2. * (mec2 + eps)**2)
        sigma += mec2 * (mec2 + 2. * eps) / (eps * (mec2 + eps)**2) * \
        math.log(eps_min_joule / (eps - eps_min_joule))
        sigma *= kappa
        sigma *= self.target.N

        return sigma

    def ejectedEnergy(self, vi, nnew):
        """
        Use Moller approximation, or delegate calculation to rsoopic (return
        result from call to rsoopic function h2crosssections.ejectedEnergy)
        """
        if self.useMollerApproximation:
            vi = vi[0:nnew]  # We may be given more velocities than we actually need
            gamma_inc = 1. / np.sqrt(1. - (vi / clight)**2)
            impactEnergy = (gamma_inc - 1.) * self.emassEV
            W = np.empty((nnew))
            for i in range(nnew):
                Xrand = random.random()
                W[i] = impactEnergy[i] * self.eps_min / \
                (impactEnergy[i] - Xrand * (impactEnergy[i] - 2 * self.eps_min))
            return W
        else:
            W = h2crosssections.ejectedEnergy(vi, nnew)
            return W[0:nnew]

    def generateAngle(self, nnew, emitted_energy, incident_energy):
        """
        Use Moller approximation, or delegate calculation to rsoopic (return
        result from call to rsoopic function h2crosssections.generateAngle)
        """
        if self.useMollerApproximation:
            theta = np.empty((nnew))
            for i in range(nnew):
                costheta = emitted_energy[i] * (incident_energy[i] + 2. * self.emassEV)
                costheta /= incident_energy[i] * (emitted_energy[i] + 2. * self.emassEV)
                costheta = math.sqrt(costheta)
                theta[i] = math.acos(costheta)
            return theta
        else:
            return h2crosssections.generateAngle(nnew, emitted_energy, incident_energy)

class H2IonizationEvent(IonizationEvent):
    """
    Derived class using the RBEB cross section (Eq. 22 in Ref. [1]),
    with parameters set for H2 (hydrogen gas)
    """
    def __init__(self):
        super(H2IonizationEvent, self).__init__(self)
        self.useMollerApproximation = False
        self.target.N = 2 # number of electrons in target (H2)
        self.target.I = 15.42593 # threshold ionization energy (in eV), from the NIST Standard Reference Database (via NIST Chemistry WebBook)
        self.target.U = 15.98 # average orbital kinetic energy (in eV), value taken from https://physics.nist.gov/PhysRefData/Ionization/intro.html

    def getCrossSection(self, vi):
        """
        Compute and return the total cross-section for electron impact ionization
        (in units of m**2)
        vi - incident electron velocity (in units of m/s, this is passed in from
        warp as vxi = uxi * gaminvi)
        Cross section sigma is given by Eq. 22 in Ref. [1]
        """
        t = h2crosssections.normalizedKineticEnergy(vi)
        if t <= 1:
            return 0.

        beta = lambda E: math.sqrt(1. - 1. / (1. + E / self.emassEV)**2)

        # initialize needed variables
        beta_t = vi / clight
        beta_u = beta(self.target.U)
        beta_b = beta(self.target.I)
        bprime = self.target.I / self.emassEV
        tprime = t * bprime
        # now add terms to sigma, one by one:
        sigma = math.log(beta_t**2 / (1. - beta_t**2)) - beta_t**2 - math.log(2. * bprime)
        sigma *= .5 * (1. - 1. / (t * t))
        sigma += 1. - 1. / t - math.log(t) / (t + 1.) * (1. + 2. * tprime) / (1. + .5 * tprime)**2
        sigma += bprime**2 / (1. + .5 * tprime)**2 * (t - 1) / 2.
        sigma *= 4. * np.pi * self.a_0**2 * fine_structure**4 * self.target.N / (beta_t**2 + beta_u**2 + beta_b**2) / (2. * bprime)

        return sigma

class IonIonizationEvent(IonizationEvent):
    """
    Base class for ion-impact ionization events. Uses the "Rayzer rule"
    (p. 514 in Ref. [5], and also last paragraph of section III.B in Ref. [4])
    to transform electron-impact ionization cross section into corresponding
    cross section for any incident ion.
    """
    def __init__(self, pm = pmass, pq = echarge, trgN = 1, trgm = None, trgq = 0.):
        super(IonIonizationEvent, self).__init__(self)
        self.projectile.mass = pm
        self.projectile.charge = pq
        target.self.N = trgN
        if trgm is not None: target.self.mass = trgm
        if trgq != 0.: target.self.charge = trgq

    def getCrossSection(self, vi):
        """
        Transform incident velocity vi using Rayzer rule, get electron
        cross section from super class, and scale it for incident ion charge
        """
        # Scale up ion velocity to that of electron with same kinetic energy
        vi *= math.sqrt(self.projectile.mass / emass)
        # Get electron-impact ionization cross section from super-class method
        sigma = super(IonIonizationEvent, self).getCrossSection(vi)
        # Scale cross section with charge
        sigma *= (self.projectile.charge / echarge)**2

        return sigma

    def ejectedEnergy(self, vi, nnew):
        """
        Transform incident velocity vi using Rayzer rule, then delegate to
        super-class method
        """
        # Scale up ion velocity to that of electron with same kinetic energy
        vi *= math.sqrt(self.projectile.mass / emass)
        return super(IonIonizationEvent, self).ejectedEnergy(vi, nnew)
