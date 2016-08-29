#!/usr/bin/env python
"""
Utility module for calculating ionization cross-sections, a la:

[1] M. E. Rudd, "Differential and total cross sections for ionization of helium and
    hydrogen by electrons," Physical Review A, vol. 44, no. 3, pp. 1644-1652,
    08/01/, 1991. http://dx.doi.org/10.1103/PhysRevA.44.1644

[2] M. E. Rudd et al., "Doubly differential electron-production cross sections
    for 200-1500-eV e-+ H2 collisions," Physical Review A, vol. 47, no. 3,
    pp. 1866, 1993. http://dx.doi.org/10.1103/PhysRevA.47.1866
"""
from __future__ import division
import time
import numpy as np
# any import from warp will trigger arg parsing, which will fail without this
# in a notebook context (or anything else with non-warp commandline args)
import warpoptions
warpoptions.ignoreUnknownArgs = True
from warp import emass, clight, jperev

bohr_radius = 5.29177e-11  # Bohr Radius
I = 15.42593  # Threshold ionization energy (in eV), from the NIST Standard Reference Database (via NIST Chemistry WebBook)
R = 13.60569  # Rydberg energy (in eV)
S = 4 * np.pi * bohr_radius**2 * (R/I)**2
fitparametern = 2.4  # species-dependent fitting parameter


def F(t):
    # Parameters for H2 specific to this fit
    a1 = 0.74
    a2 = 0.87
    a3 = -0.60
    return np.divide(1, t) * (a1 * np.log(t) + a2 + np.divide(1, t) * a3)


def f_1(w, t, n=fitparametern):
    return np.divide(1, (w+1)**n) + np.divide(1, (t-w)**n) - np.divide(1, ((w+1) * (t-w))**(n/2))


def normalizedKineticEnergy(vi=None):
    """
    Compute the normalized kinetic energy n = T/I given an input velocity
    """
    gamma_in = 1. / np.sqrt(1 - (vi/clight)**2)
    T = (gamma_in - 1) * emass * clight**2 / jperev  # kinetic energy (in eV) of incident electron
    t = T / I  # normalized kinetic energy
    return t


def h2_ioniz_ddcs(vi=None, vo=None, ke_emitted=None):
    """
    Compute the doubly-differential cross-section for impact ionization of H2 by e-
    vi - incident electron velocity in m/s; this is passed in from warp as vxi=uxi*gaminvi etc.
    vo - ejected electron velocity in m/s
    ke_emitted - kinetic energy (in eV) of emitted secondary electron
    """
    pass
    # t = normalizedKineticEnergy(vi)
    # w = ke_emitted / I
    # sigma = G1 * (f_BE + G4*f_b)
    # return np.nan_to_num(sigma)


def h2_ioniz_sdcs(vi=None, vo=None):
    """
    Compute the differential cross-section for impact ionization of H2 by e-, per David's formula
    vi - incident electron velocity in m/s; this is passed in from warp as vxi=uxi*gaminvi etc.
    vo - velocity in m/s of emitted electron
    """
    t = normalizedKineticEnergy(vi)
    w = normalizedKineticEnergy(vo)
    sigma = S * F(t) * f_1(w, t) / I
    return np.nan_to_num(sigma)


def h2_ioniz_crosssection(vi=None):
    """
    Compute the total cross-section for impact ionization of H2 by e-, per David's formula
    vi - incident electron velocity in m/s; this is passed in from warp as vxi=uxi*gaminvi etc.
    """
    t = normalizedKineticEnergy(vi)
    n = fitparametern

    def g1(t, n):
        return (1 - t**(1-n)) / (n-1) - (2 / (t+1))**(n/2) * (1 - t**(1 - n/2)) / (n-2)

    sigma = S * F(t) * g1(t, n)
    return np.nan_to_num(sigma)


def ejectedEnergy(vi, nnew):
    """
    Selection of an ejected electron energy (in eV) based on [1], adapted from
    XOOPIC's MCCPackage::ejectedEnergy routine
    """
    vi = vi[0:nnew]  # We may be given more velocities than we actually need

    tstart = time.time()
    emassEV = emass*clight**2 / jperev
    gamma_inc = 1/np.sqrt(1-(vi/clight)**2)
    impactEnergy = (gamma_inc-1) * emassEV

    tPlusMC = impactEnergy + emassEV
    twoTplusMC = impactEnergy + tPlusMC
    tPlusI = impactEnergy + I
    tMinusI = impactEnergy - I

    invTplusMCsq = 1. / (tPlusMC * tPlusMC)
    tPlusIsq = tPlusI ** 2
    iOverT = I / impactEnergy

    funcT1 = 14./3. + .25 * tPlusIsq * invTplusMCsq - emassEV * twoTplusMC * iOverT * invTplusMCsq
    funcT2 = 5./3. - iOverT - 2.*iOverT*iOverT/3. + .5 * I * tMinusI * invTplusMCsq + emassEV*twoTplusMC*I*invTplusMCsq*np.log(iOverT)/tPlusI

    aGreaterThan = funcT1 * tMinusI / funcT2 / tPlusI

    needToTryAgain = True
    npart = nnew  # number of particles to generate on each loop
    frand = np.random.uniform

    wOut = np.array([])

    while len(wOut) < nnew:
        # print("%i particles left to generate for" % npart)

        # wTest is the inverse of F(W)
        # The random number, called F(W) in the notes and randomFW here,
        #   is the antiderivative of a phony_but_simple probability, which
        #   is always >= the correct_but_messy probability.

        randomFW = frand(size=nnew) * aGreaterThan

        # wTest is the inverse of F(W)
        wTest = I*funcT2*randomFW/(funcT1-funcT2*randomFW)

        # Because we are not working directly with the distribution function,
        #   we must use the "rejection method".  This involves generating
        #   another random number and seeing whether it is > the ratio of
        #   the true probability over the phony_but_simple probability.

        wPlusI       = wTest + I
        wPlusIsq     = wPlusI * wPlusI
        invTminusW   = 1./(impactEnergy-wTest)
        invTminusWsq = invTminusW**2
        invTminusW3  = invTminusW**3

        probabilityRatio = (1. + 4.*I/wPlusI/3. + wPlusIsq*invTminusWsq + 4.*I*wPlusIsq*invTminusW3/3. - emassEV*twoTplusMC*wPlusI*invTminusW*invTplusMCsq + wPlusIsq*invTplusMCsq) / funcT1

        mask = (probabilityRatio >= frand(size=nnew))
        # npart -= np.sum(mask)  # Decrement by the number of passing particles

        # Append the energies that meet the selection criterion
        wOut = np.append(wOut, wTest[mask])

    print("Spent %.3f s generating ejected energies" % (time.time()-tstart))
    return wOut[0:nnew]  # Might possibly have more particles than necessary, but should have at least that many
