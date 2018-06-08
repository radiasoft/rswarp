#!/usr/bin/env python
"""
Ionization class derived from Warp's Ionization, with some improvements.
"""

from warp import *
from warp.particles import ionization
from rswarp.utilities.beam_manipulation import rotateVec
import numpy as np
import time
import types
from shutil import os
import h5py

__all__ = ['Ionization']


def tryFunctionalForm(f, *args, **kwargs):
    try:
        res = f(*args, **kwargs)
    except TypeError:
        res = f
    return res


class Ionization(ionization.Ionization):
    """
    Extension of Warp's Ionization class including provisions for more detailed ionization physics.
    """
    def generateEmittedVelocity(self, nnew, emitted_energy0, emitted_energy_sigma=None):
        """
        Generate array of emitted particle velocities

        nnew - number of new particles
        emitted_energy0 - central energy of emission distribution (in eV)
        emitted_energy_sigma - energy spread of emission distribution (in eV)
        theta - angle with
        """
        if emitted_energy_sigma > 0:
            ek = random.normal(0., emitted_energy_sigma, nnew)
        else:
            ek = zeros(nnew)
        ek = abs(ek + emitted_energy0)  # kinetic energy

        # fraction of kinetic energy in the transverse direction
        frac_trans = ranf(ek)
        frac_long = 1.0 - frac_trans

        fact = jperev / (emass * clight**2)

        ek_trans = frac_trans * ek
        ek_long = frac_long * ek
        gamma_trans = ek_trans * fact + 1
        gamma_long = ek_long * fact + 1
        u_trans = clight * sqrt(1 - 1 / gamma_trans**2)
        u_long = np.sign(2. * ranf(ek) - 1) * clight * \
            sqrt(1 - 1 / gamma_long**2)

        # velocity direction: random in (x-y) plane
        phi = 2. * pi * ranf(ek)

        # find components of v*gamma:
        uxnew = u_trans * cos(phi)
        uynew = u_trans * sin(phi)
        uznew = u_long
        return [uxnew, uynew, uznew]

    def scale_primary_momenta(self, incident_species, ipg, emitted_energy0,
                              emitted_energy_sigma, i1, i2, io):
        m = incident_species.mass
        Erest = m*clight**2
        uxi = ipg.uxp[i1:i2:self.stride][io]
        uyi = ipg.uyp[i1:i2:self.stride][io]
        uzi = ipg.uzp[i1:i2:self.stride][io]
        gaminviold = ipg.gaminv[i1:i2:self.stride][io]
        Eold = Erest * 1./gaminviold
        Enew = Eold - (emitted_energy0 + (emitted_energy_sigma or 0)) * jperev
        if np.any(Enew < Erest):
            print("WARNING: requested emitted_energy0 would reduce energy of {} "
                  "particles to less than rest mass, clamping incident energy "
                  "to rest mass.".format(np.sum(Enew < Erest))
                  )
        Enew = np.maximum(Erest, Enew)
        gaminvinew = np.minimum(Erest / Enew, 1.)
        assert(np.all(gaminvinew <= 1)), "Gamma should be <= than 1"
        unew = clight * sqrt(1 - gaminvinew**2) / gaminvinew
        uold = sqrt(uxi**2 + uyi**2 + uzi**2)
        uscale = unew / uold
        return uscale

    def add(self, incident_species, emitted_species, cross_section=None,
            target_species=None, ndens=None, target_fluidvel=None,
            emitted_energy0=None, emitted_energy_sigma=None,
            incident_pgroup=top.pgroup, target_pgroup=top.pgroup, emitted_pgroup=top.pgroup,
            l_remove_incident=None, l_remove_target=None, emitted_tag=None,
            sampleIncidentAngle=None, sampleEmittedAngle=None, writeAngleDataDir=None,
            writeAnglePeriod=100):
        """
        Add a particular class of ionization event to the simulation.  Any given
        ionization event looks like:

        incident + target -> incident(?) + target(?) + emitted

        Where the (?) events are controlled by l_remove_incident and l_remove_target

        emitted_energy0 - may be a callable returning a numpy array of cetnral
            energies (in eV) or a scalar value (or an array of length
            len(emitted_species) of the same)
        emitted_energy_sigma - the same as emitted_energy0, but for distribution width
        sampleIncidentAngle - callable (or array of len(emitted_species)) for
            sampling scattering angles of incident species
        sampleIncidentAngle - callable (or array of len(emitted_species)) for
            sampling scattering angles of emitted species
        writeAngleDataDir - directory used to write emission angle diagnostics
        writeAnglePeriod - period to write out angle diagnostics
        """
        # Do all of Warp's normal initialization
        if not iterable(emitted_species):
            emitted_species = [emitted_species]
        for i in range(len(emitted_species)):
            if emitted_species[i].sw != emitted_species[i - 1].sw:
                print("WARNING: Not all emitted species have the same weight\n"
                      "         For incident species: {}".format(incident_species.name))
                print("         Using weighting of species: {}".format(emitted_species[0].name))
                break

        ionization.Ionization.add(self, incident_species, emitted_species, cross_section,
                                  target_species, ndens, target_fluidvel,
                                  emitted_energy0, emitted_energy_sigma,
                                  incident_pgroup, target_pgroup, emitted_pgroup,
                                  l_remove_incident, l_remove_target, emitted_tag
                                  )

        # Extended class-specific operations following Warp's initialization
        self.sampleIncidentAngle = sampleIncidentAngle
        self.sampleEmittedAngle = sampleEmittedAngle
        self.writeAngleDataDir = writeAngleDataDir
        self.writeAnglePeriod = writeAnglePeriod

        if self.writeAngleDataDir and not os.path.exists(self.writeAngleDataDir):
            os.makedirs(self.writeAngleDataDir)

    def generate(self, dt=None):
        self.init_grid()
        if dt is None:
            dt = top.dt
        if self.l_timing:
            t1 = time.clock()
        for target_species in self.target_dens:
            self.target_dens[target_species]['ndens_updated'] = 0
        for incident_species in self.inter:
            npinc = 0
            ispushed = 0
            ipg = self.inter[incident_species]['incident_pgroup']
            tpg = self.inter[incident_species]['target_pgroup']
            epg = self.inter[incident_species]['emitted_pgroup']
            for js in incident_species.jslist:
                npinc += ipg.nps[js]
                if ipg.ldts[js]:
                    ispushed = 1
            if npinc == 0 or not ispushed:
                continue
            for it, target_species in enumerate(self.inter[incident_species]['target_species']):
                ndens = self.inter[incident_species]['ndens'][it]
                target_fluidvel = self.inter[
                    incident_species]['target_fluidvel'][it]
                if ndens is not None:
                    continue
                else:
                    if self.target_dens[target_species]['ndens_updated']:
                        continue
                    else:
                        self.target_dens[target_species]['ndens_updated'] = 1
                    ndens = self.target_dens[target_species]['ndens']
                    target_fluidvel = self.target_dens[
                        target_species]['target_fluidvel']
                    nptarget = 0
                    for jstarget in target_species.jslist:
                        nptarget += tpg.nps[jstarget]
                    if nptarget == 0:
                        continue
                    self.ndensc[...] = 0.
                    ndens[...] = 0.

                    for jstarget in target_species.jslist:
                        self.depositTargetSpecies(jstarget)

        for incident_species in self.inter:
            npinc = 0
            ispushed = 0
            ipg = self.inter[incident_species]['incident_pgroup']
            epg = self.inter[incident_species]['emitted_pgroup']
            for js in incident_species.jslist:
                npinc += ipg.nps[js]
                if ipg.ldts[js]:
                    ispushed = 1
            if npinc == 0 or not ispushed:
                continue
            for it, target_species in enumerate(self.inter[incident_species]['target_species']):
                ndens = self.inter[incident_species]['ndens'][it]
                target_fluidvel = self.inter[
                    incident_species]['target_fluidvel'][it]
                for js in incident_species.jslist:
                    i1 = ipg.ins[js] - 1 + top.it % self.stride
                    i2 = ipg.ins[js] + ipg.nps[js] - 1
                    xi = ipg.xp[i1:i2:self.stride]  # .copy()
                    yi = ipg.yp[i1:i2:self.stride]  # .copy()
                    zi = ipg.zp[i1:i2:self.stride]  # .copy()
                    ni = shape(xi)[0]
                    gaminvi = ipg.gaminv[i1:i2:self.stride]  # .copy()
                    uxi = ipg.uxp[i1:i2:self.stride]  # .copy()
                    uyi = ipg.uyp[i1:i2:self.stride]  # .copy()
                    uzi = ipg.uzp[i1:i2:self.stride]  # .copy()
                    if top.wpid > 0:
                        # --- Save the wpid of the incident particles so that it can be
                        # --- passed to the emitted particles.
                        wi = ipg.pid[i1:i2:self.stride, top.wpid - 1]
                    else:
                        wi = 1.
                    if top.injdatapid > 0:
                        # --- Save the injdatapid of the incident particles so that it can be
                        # --- passed to the emitted particles.
                        injdatapid = ipg.pid[
                            i1:i2:self.stride, top.injdatapid - 1]
                    else:
                        injdatapid = None
                    # --- get velocity in lab frame if using a boosted frame of reference
                    if top.boost_gamma > 1.:
                        uzboost = clight * sqrt(top.boost_gamma**2 - 1.)
                        setu_in_uzboosted_frame3d(ni, uxi, uyi, uzi, gaminvi,
                                                  -uzboost,
                                                  top.boost_gamma)
                    vxi = uxi * gaminvi
                    vyi = uyi * gaminvi
                    vzi = uzi * gaminvi
                    # --- get local target density
                    if ndens is None:
                        ndens = self.target_dens[target_species]['ndens']
                    if isinstance(ndens, (types.IntType, float)):
                        dp = ones(ni, 'd') * ndens
                        if target_fluidvel is None:
                            xmin = self.xmin
                            xmax = self.xmax
                            ymin = self.ymin
                            ymax = self.ymax
                            zmin = self.zmin
                            zmax = self.zmax
                        else:
                            vxtf = target_fluidvel[0]
                            vytf = target_fluidvel[1]
                            vztf = target_fluidvel[2]
                            xmin = self.xmin + vxtf * top.time
                            xmax = self.xmax + vxtf * top.time
                            ymin = self.ymin + vytf * top.time
                            ymax = self.ymax + vytf * top.time
                            zmin = self.zmin + vztf * top.time
                            zmax = self.zmax + vztf * top.time
                        if w3d.solvergeom == w3d.RZgeom:
                            ri = sqrt(xi * xi + yi * yi)
                            dp = where((ri >= xmin) & (ri <= xmax) &
                                       (zi >= zmin) & (zi <= zmax), dp, 0.)
                        else:
                            dp = where((xi >= xmin) & (xi <= xmax) &
                                       (yi >= ymin) & (yi <= ymax) &
                                       (zi >= zmin) & (zi <= zmax), dp, 0.)
                    else:
                        dp = zeros(ni, 'd')
                        getgrid3d(ni, xi, yi, zi, dp,
                                  self.nx, self.ny, self.nz, ndens,
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)
                    # --- get local target fluid velocity
                    if target_fluidvel is None:
                        if target_species is None:
                            target_fluidvel = [0., 0., 0.]
                        else:
                            target_fluidvel = self.target_dens[
                                target_species]['target_fluidvel']
                    if isinstance(target_fluidvel, list):
                        vxtf = target_fluidvel[0]
                        vytf = target_fluidvel[1]
                        vztf = target_fluidvel[2]
                    else:
                        vxtf = zeros(ni, 'd')
                        vytf = zeros(ni, 'd')
                        vztf = zeros(ni, 'd')
                        getgrid3d(ni, xi, yi, zi, vxtf,
                                  self.nx, self.ny, self.nz, target_fluidvel[
                                      ..., 0],
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)
                        getgrid3d(ni, xi, yi, zi, vytf,
                                  self.nx, self.ny, self.nz, target_fluidvel[
                                      ..., 1],
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)
                        getgrid3d(ni, xi, yi, zi, vztf,
                                  self.nx, self.ny, self.nz, target_fluidvel[
                                      ..., 2],
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)

                    # compute the relative velocity
                    # NOTE that at this point, the target species is assumed to have a negligible velocity.
                    # this needs to be modified if this approximation is not
                    # valid.
                    vxr = vxi - vxtf
                    vyr = vyi - vytf
                    vzr = vzi - vztf
                    vi = sqrt(vxr * vxr + vyr * vyr + vzr * vzr)
                    cross_section = self.getcross_section(
                        self.inter[incident_species]['cross_section'][it], vi)

                    # Number of collisions
                    ncol = dp * cross_section * vi * dt * ipg.ndts[js] * \
                        self.stride / self.inter[incident_species]['emitted_species'][0][0].sw * \
                        ipg.sw[js]
                    if top.boost_gamma > 1.:
                        ncol *= top.gammabar_lab / top.gammabar

                    # If the incident species is being removed, then only one collision event can happen.
                    # Otherwise, a single particle may collide multiple times in a time step if ncol > 1.
                    if self.inter[incident_species]['remove_incident'][it]:
                        # Note that ncol is being set to slightly less than one. In the code below, adding ranf
                        # to it will guarantee that it will end up as one. (Is this needed?)
                        ncol = where(ncol >= 1., 1. - 1.e-10, ncol)

                    # Get a count of the number of collisions for each particle. A random number is added to
                    # ncol so that a fractional value has chance to result in a collision.
                    ncoli = aint(ncol + ranf(ncol))

                    # --- Select the particles that will collide
                    io = compress(ncoli > 0, arange(ni))
                    nnew = len(io)

                    if None in self.inter[incident_species]['emitted_energy0'][it]:
                        # When emitted_energy0 is not specified, use the velocity of
                        # the incident particles for the emitted particles.
                        uxnewsave = uxi
                        uynewsave = uyi
                        uznewsave = uzi

                    if self.inter[incident_species]['remove_incident'][it]:
                        # If projectile is modified, then need to delete it
                        put(ipg.gaminv, array(io) * self.stride + i1, 0.)

                    # The position of the incident particle is at or near the incident particle
                    xnew = xi
                    ynew = yi
                    znew = zi

                    incidentvelocities = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    originalvelocities = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    emissionangles = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    emittedvelocities = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    recoilangles = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}

                    # Loop until there are no more collision events that need handling
                    while nnew > 0:

                        # The emitted particles positions, in some cases, are slightly
                        # offset from the incident
                        xnewp = xnew[io]
                        ynewp = ynew[io]
                        znewp = znew[io]
                        xnew = xnewp + (ranf(xnewp) - 0.5) * 1.e-10 * self.dx
                        ynew = ynewp + (ranf(ynewp) - 0.5) * 1.e-10 * self.dy
                        znew = znewp + (ranf(znewp) - 0.5) * 1.e-10 * self.dz
                        if top.wpid == 0:
                            w = 1.
                        else:
                            w = wi[io]

                        # The injdatapid value needs to be copied to the emitted particles
                        # so that they are handled properly in the region near the source.
                        if top.injdatapid > 0:
                            injdatapid = injdatapid[io]

                        # If the emitted energy was not specified, the emitted particle will be
                        # given the same velocity of the incident particle.
                        # TODO: Not sure if the None case for emitted_energy0 is properly handled
                        if None in self.inter[incident_species]['emitted_energy0'][it]:
                            uxnewsave = uxnewsave[io]
                            uynewsave = uynewsave[io]
                            uznewsave = uznewsave[io]

                        # For each collision calculate product velocity components and modify incident particle
                        for ie, emitted_species in enumerate(self.inter[incident_species]['emitted_species'][it]):

                            incident_ke = (incident_species.mass*clight**2)/jperev * (1. / ipg.gaminv[io * self.stride + i1] - 1.)

                            # If no emitted_energy0 then emission particle velocity set to incident particle velocity
                            if self.inter[incident_species]['emitted_energy0'][it][ie] is not None:
                                # --- Create new velocities for the emitted particles.
                                emitted_energy0 = tryFunctionalForm(self.inter[incident_species]['emitted_energy0'][it][ie], vi=vi, nnew=nnew)
                                emitted_energy_sigma = tryFunctionalForm(self.inter[incident_species]['emitted_energy_sigma'][it][ie], vi=vi, nnew=nnew)
                                if emitted_energy_sigma is None:
                                    emitted_energy_sigma = 0
                                else:
                                    emitted_energy = np.abs(emitted_energy0 +
                                                            np.random.normal(loc=0., scale=emitted_energy_sigma, size=nnew))
                            else:
                                uxnew = uxnewsave
                                uynew = uynewsave
                                uznew = uznewsave

                            ui = np.vstack((uxi[io], uyi[io], uzi[io])).T
                            incidentvelocities[emitted_species] = np.append(incidentvelocities[emitted_species], ui)

                            # Initial calculation of emitted particle velocity components
                            gnew = 1. + emitted_energy*jperev / (emitted_species.mass * clight**2)
                            bnew = np.sqrt(1 - 1/gnew**2)
                            norm = np.linalg.norm(ui, axis=1)
                            uxnew = uxi[io]/norm * bnew*gnew*clight
                            uynew = uyi[io]/norm * bnew*gnew*clight
                            uznew = uzi[io]/norm * bnew*gnew*clight
                            # Remove energy from incident particle
                            scale = self.scale_primary_momenta(incident_species, ipg,
                                                               emitted_energy0, emitted_energy_sigma, i1, i2, io)

                            uxi[io] *= scale
                            uyi[io] *= scale
                            uzi[io] *= scale

                            assert np.all(vzi != 0), "Not all components of vzi are non-zero"

                            v1 = np.vstack((vxi, vyi, vzi)).T[io]
                            v2 = v1.copy()
                            v2[:, 2] = 0
                            rotvec = np.cross(v1, v2)

                            if hasattr(self.sampleEmittedAngle, '__call__'):
                                uemit = np.vstack((uxnew, uynew, uznew)).T
                                angles = np.array(self.sampleEmittedAngle(nnew=nnew, emitted_energy=emitted_energy, incident_energy=incident_ke))
                                if self.writeAngleDataDir and top.it % self.writeAnglePeriod == 0:
                                    originalvelocities[emitted_species] = np.append(originalvelocities[emitted_species], uemit)
                                    emissionangles[emitted_species] = np.append(emissionangles[emitted_species], angles)

                                # Altitude
                                uxnew, uynew, uznew = [l.flatten() for l in rotateVec(vec=uemit, rotaxis=rotvec, theta=angles)]
                                uemit = np.vstack((uxnew, uynew, uznew)).T

                                # Azimuth (random)
                                uxnew, uynew, uznew = [l.flatten() for l in rotateVec(vec=uemit, rotaxis=v1, theta=np.random.uniform(size=uemit.shape[0])*2*np.pi)]
                                uemit = np.vstack((uxnew, uynew, uznew)).T

                                if self.writeAngleDataDir and top.it % self.writeAnglePeriod == 0:
                                    emittedvelocities[emitted_species] = np.append(emittedvelocities[emitted_species], uemit)

                                if hasattr(self.sampleIncidentAngle, '__call__'):
                                    rangles = np.array(self.sampleIncidentAngle(nnew=nnew, emitted_energy=emitted_energy0, incident_energy=incident_ke, emitted_theta=emissionangles))
                                    if self.writeAngleDataDir and top.it % self.writeAnglePeriod == 0:
                                        recoilangles[emitted_species] = np.append(recoilangles[emitted_species], rangles)
                                    vin = np.vstack((vxi[io], vyi[io], vzi[io])).T
                                    vxi[io], vyi[io], vzi[io] = [l.flatten() for l in rotateVec(vec=vin, rotaxis=rotvec, theta=rangles)]

                            ginew = 1. / sqrt(1. + (uxnew**2 + uynew ** 2 + uznew**2) / clight**2)
                            # get velocity in boosted frame if using a boosted frame of reference
                            if top.boost_gamma > 1.:
                                setu_in_uzboosted_frame3d(shape(ginew)[0], uxnew, uynew, uznew, ginew,
                                                          uzboost,
                                                          top.boost_gamma)

                            if self.l_verbose:
                                print 'add ', nnew, emitted_species.name, ' from by impact ionization:', incident_species.name, '+', ((target_species is None and 'background gas') or target_species.name)
                            if self.inter[incident_species]['remove_incident'][it] and (emitted_species.type is incident_species.type):
                                self.addpart(nnew, xnewp, ynewp, znewp, uxnew, uynew, uznew, ginew, epg, emitted_species.jslist[0],
                                             self.inter[incident_species]['emitted_tag'][it], injdatapid, w)
                            else:
                                self.addpart(nnew, xnew, ynew, znew, uxnew, uynew, uznew, ginew, epg, emitted_species.jslist[0],
                                             self.inter[incident_species]['emitted_tag'][it], injdatapid, w)
                        ncoli = ncoli[io] - 1
                        io = arange(nnew)[ncoli > 0]
                        nnew = len(io)
                    try:
                        rank = comm_world.rank
                    except NameError:
                        rank = 0
                    if rank == 0:
                        if self.writeAngleDataDir and top.it % self.writeAnglePeriod == 0:
                            for em in self.inter[incident_species]['emitted_species'][it]:
                                fmt = (int(top.it), incident_species.name, em.name)
                                start = time.time()
                                with h5py.File(os.path.join(self.writeAngleDataDir, 'angledata.h5'), 'a') as f:
                                    f['data/{}/{}/{}/incidentvelocities'.format(*fmt)] = incidentvelocities[em]
                                    f['data/{}/{}/{}/originalvelocities'.format(*fmt)] = originalvelocities[em]
                                    f['data/{}/{}/{}/emittedvelocities'.format(*fmt)] = emittedvelocities[em]
                                    f['data/{}/{}/{}/emissionangles'.format(*fmt)] = emissionangles[em]
                                    f['data/{}/{}/{}/recoilangles'.format(*fmt)] = recoilangles[em]
                                print("Spent {} ms on hdf5 writing".format((time.time() - start)*1000))

        # make sure that all particles are added and cleared
        for pg in self.x:
            for js in self.x[pg]:
                self.flushpart(pg, js)

        for incident_species in self.inter:
            for js in incident_species.jslist:
                processlostpart(top.pgroup, js + 1,
                                top.clearlostpart, top.time, top.zbeam)

                if self.l_timing:
                    print 'time ionization = ', time.clock() - t1, 's'

    def depositTargetSpecies(self, jstarget):
        """ Depositing target species to the grid """
        tpg = self.inter[incident_species]['target_pgroup']
        i1 = tpg.ins[jstarget] - 1  # ins is index of first member of species
        i2 = tpg.ins[jstarget] + tpg.nps[jstarget] - 1
        xt = tpg.xp[i1:i2]
        yt = tpg.yp[i1:i2]
        zt = tpg.zp[i1:i2]
        git = tpg.gaminv[i1:i2]
        vxt = tpg.uxp[i1:i2] * git
        vyt = tpg.uyp[i1:i2] * git
        vzt = tpg.uzp[i1:i2] * git
        fact = 1.
        if w3d.l4symtry:
            xt = abs(xt)
            fact = 0.25
        elif w3d.l2symtry:
            fact = 0.5
        if w3d.l2symtry or w3d.l4symtry:
            yt = abs(yt)
        if top.wpid == 0:
            weights = ones(tpg.nps[jstarget], 'd')
        else:
            weights = tpg.pid[i1:i2, top.wpid - 1]
        # --- deposit density
        deposgrid3d(1, tpg.nps[jstarget], xt, yt, zt,
                    tpg.sw[jstarget] * self.invvol * fact * weights,
                    self.nx, self.ny, self.nz, ndens, self.ndensc,
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    self.zmin, self.zmax)
        # --- computes target fluid velocity
        deposgrid3dvect(0, tpg.nps[jstarget], xt, yt, zt, vxt, vyt, vzt,
                        tpg.sw[jstarget] * self.invvol * fact * weights,
                        self.nx, self.ny, self.nz, target_fluidvel, self.ndensc,
                        self.xmin, self.xmax, self.ymin, self.ymax,
                        self.zmin, self.zmax)
