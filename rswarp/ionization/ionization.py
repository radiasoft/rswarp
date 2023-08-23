# -*- coding: utf-8 -*-
u"""?

:copyright: Copyright (c) 2020 RadiaSoft LLC.  All Rights Reserved.
:license: http://www.apache.org/licenses/LICENSE-2.0.html
"""
from __future__ import absolute_import, division, print_function
import warpoptions
warpoptions.ignoreUnknownArgs = True
from warp import *
from warp.particles import ionization
from rswarp.utilities.beam_manipulation import rotateVec
import numpy as np
import time
import os
import h5py


_DEBUG = False
__all__ = ['Ionization']


class DummySpecies:
    def __init__(self):
        self.name = 'dummy'
        self.sw = 1.0

def _make_callable(arg):
    if callable(arg):
        return arg

    return lambda *args, **kwargs: arg


class Ionization(ionization.Ionization):
    """
    Extension of Warp's warp.particles.ionization.Ionization class
     including provisions for more detailed ionization physics.
    """
    
    def _get_volume(self):
        v = (w3d.xmmax - w3d.xmmin) * \
            (w3d.ymmax - w3d.ymmin) * \
            (w3d.zmmax - w3d.zmmin)
        return v
    
    def _add_bound(self, method):
        new_method = method.__get__(self)
        self.__setattr__(method.__name__, new_method)
        return self.__getattribute__(method.__name__)

    def _generate_emitted_velocity(self, nnew, emitted_energy0, emitted_energy_sigma=None):
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

    def _scale_primary_momenta(self, incident_species, ipg, energy, i1, i2, io):
        if _DEBUG:
            print('indicies:', io)
        m = incident_species.mass
        Erest = m*clight**2
        uxi = ipg.uxp[i1:i2:self.stride][io]
        uyi = ipg.uyp[i1:i2:self.stride][io]
        uzi = ipg.uzp[i1:i2:self.stride][io]
        if _DEBUG:
            print("vel mag:", np.sqrt(uxi**2 + uyi**2 + uzi**2))
        gaminviold = np.sqrt(1. - (uxi**2 + uyi**2 + uzi**2) / clight**2)
        # gaminviold = ipg.gaminv[i1:i2:self.stride][io]
        if _DEBUG:
            print("gammainv Eold:", gaminviold)
        Eold = Erest * 1./gaminviold
        if _DEBUG:
            print("Eold in eV:", Eold / np.abs(incident_species.charge))
        Enew = Eold - energy * jperev
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
    

    def getcross_section(self, cross_section, vi):
        """The cross section can either be a scalar or a functional that returns
        a value given a velocity. This returns the Appropriate value.
        """
        try:
            result = cross_section(self, vi)
        except TypeError:
            try:
                result = cross_section(vi)
            except TypeError:
                result = cross_section
        return result

    def add(self, incident_species, emitted_species, cross_section=None,
            target_species=None, ndens=None, reservoir=None, target_fluidvel=None,
            emitted_energy0=None, emitted_energy_sigma=None, temperature=None, conserve_energy=None,
            incident_pgroup=top.pgroup, target_pgroup=top.pgroup, emitted_pgroup=top.pgroup,
            l_remove_incident=None, l_remove_target=None, emitted_tag=None,
            sampleIncidentAngle=None, sampleEmittedAngle=None, writeAngleDataDir=None,
            writeAnglePeriod=100):
        """
        Add an ionization event to the `Ionization` instance. The structure of an event varies depending on
        what parameters are set in `add`:

        If `target_species` is set:
        incident + target -> incident(*) + target(*) + emitted
         where (*) indicates that these can be removed from the simulation in a collision

        If `ndens` is set and `target_species` is None then a gas 'reservoir' is used to calculate the 'target'
        species density instead of macroparticles and the process is:
        incident + (reservoir) -> incident(*) + emitted
         where (*) indicates that these can be removed from the simulation in a collision

        A number of parameters have different operational modes depending on the input applied. Most are specfied
        below in Args. Additionally, not setting `emitted_energy0` will cause emitted particles to be created
        with the same momentum vector as the incident particle. This is probably completely unphysical and retained
        for legacy purposes.

        When using a gas reservoir it may be desirable to attribute some thermal velocity distribution to the gas
        when the ions are created.

        For any user supplied functions that require velocity as an input it should be expected that velocity
        supplied will have units of m/s.

        Args:
            incident_species: warp.particles.species.Species instance for incident species of the ionization event
            emitted_species: warp.particles.species.Species instance or list of instances for the emitted species
                of the ionization event
            cross_section: Scalar or callable function representing the interaction cross section. Function must take
                incident velocity as input and return the cross section. Cross section is in units of m^2.
            target_species: warp.particles.species.Species instance for target species of the ionization event. Should
                not be set in ndens being used.
            ndens: Gas density of reservoir if being used. Density is in units of 1/m^3.
            target_fluidvel: (list) List containing Cartesian velocity vector components of the gas reservoir. Can be
                used to update the position of the gas reservoir in time. Velocity in m/s.
            emitted_energy0: (list) List of length len(emitted_species) with scalar or callable functions for the
                energy of each emitted particle. Functions should take (incident velocity, number of events)
                as parameters. To create species with a thermal energy distribution set to 'thermal'.
                the value of `temperature` must aslo be set. Thermal energy will not be removed from incident particles.
                Energy is in units of eV.
            emitted_energy_sigma: (list) List of length len(emitted_species) with scalar or callable functions for the
                energy spread of each emitted particle. This will be added to the value of emitted_energy0 when
                a particle is emitted. If nothing is set then this parameter defaults to 0.
                Functions should take (incident velocity, number of events)as parameters. If scalar values are set then
                they indicate the width of a gaussian distribution from which to sample the energy spread.
                Energy spread is in units of eV.
            temperature (list): List of length len(emitted_species) giving temperature in the case of species having
                a thermal velocity distribution. This is intended to account for the thermal motion of the background
                gas in cases where a gas reservior is being used. Temperature value is ignored unless the species
                has emitted_energy0[species] = 'thermal. Temperature in units of Kelvin.
            incident_pgroup: Default - top.pgroup. `ParticleGroup` instance containing incident `Species`.
            target_pgroup: Default - top.pgroup. `ParticleGroup` instance containing target `Species`.
            emitted_pgroup: Default - top.pgroup. `ParticleGroup` instance containing emitted `Species`.
            l_remove_incident: Incident macroparticles in ionization event will be removed from the simulation.
            l_remove_target: Target macroparticles in ionization event will be removed from the simulation.
            emitted_tag: Tag for emitted species.
            sampleIncidentAngle: Callable function (or array of len(emitted_species)) for sampling scattering angles
                of incident species.
            sampleEmittedAngle: Callable function (or array of len(emitted_species)) for sampling scattering angles
                of emitted species.
            writeAngleDataDir: Directory to write HDF5 files with incident and emission angle data. Optional.
            writeAnglePeriod: Default - 100. Step period to write out angle data.
        
        Attributes:
            _incident_species: Current incident species in generate loop
            _target_species: Current target species in generate loop
            io: Indices of colliding macroparticles in _incident_species
            xi, yi, zi: Incident particle positions
            uxi, uyi, uzi: Incident particle momenta
        
        Returns:
                None
        """
        # Do all of Warp's normal initialization
        if not iterable(emitted_species):
            # Ionization.add checks lengths of emitted_energy0 and emitted_energy_sigma
            emitted_species = [emitted_species]

        for i in range(len(emitted_species)):
            if emitted_species[i].sw != emitted_species[i - 1].sw:
                print("WARNING: Not all emitted species have the same weight\n"
                      "         For incident species: {}".format(incident_species.name))
                print("         Using weighting of species: {}".format(emitted_species[0].name))
                break

        self.reservoir = reservoir
        if reservoir:
            if ndens:
                print("ndens and reservoir given to Ionization setup. reservoir will be used.")
            ndens = self.reservoir.get_density()

        ionization.Ionization.add(self, incident_species, emitted_species, cross_section,
                                  target_species, ndens, target_fluidvel,
                                  emitted_energy0, emitted_energy_sigma,
                                  incident_pgroup, target_pgroup, emitted_pgroup,
                                  l_remove_incident, l_remove_target, emitted_tag
                                  )

        # Extended class-specific operations following Warp's initialization
        for i in range(len(self.inter[incident_species]['emitted_energy0'])):
            for j, (ee, ees) in enumerate(zip(self.inter[incident_species]['emitted_energy0'][i],
                                            self.inter[incident_species]['emitted_energy_sigma'][i])):
                # bound_method = self._add_bound(_make_callable(ee))
                self.inter[incident_species]['emitted_energy0'][i][j] = _make_callable(ee)
                # bound_method = self._add_bound(_make_callable(ees))
                self.inter[incident_species]['emitted_energy_sigma'][i][j] = _make_callable(ees)

        self.sampleIncidentAngle = sampleIncidentAngle
        self.sampleEmittedAngle = sampleEmittedAngle
        self.writeAngleDataDir = writeAngleDataDir
        self.writeAnglePeriod = writeAnglePeriod
        self.inter[incident_species]['thermal'] = []
        self.inter[incident_species]['temperature'] = []
        self.inter[incident_species]['conserve_energy'] = []
        self.inter[incident_species]['thermal'].append([energy == 'thermal' for energy in emitted_energy0])
        if conserve_energy:
            self.inter[incident_species]['conserve_energy'].append([conserve for conserve in conserve_energy])
        else:
            self.inter[incident_species]['conserve_energy'].append([True, ]*len(emitted_species))
        self.inter[incident_species]['temperature'].append(temperature)
        for i, ip in enumerate(self.inter[incident_species]['thermal']):
            for j, ep in enumerate(ip):
                if ep:
                    self.inter[incident_species]['emitted_energy0'][i][j] = 0

        if self.writeAngleDataDir and not os.path.exists(self.writeAngleDataDir):
            # TODO: Change to only write file on head process
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
                    if self.reservoir and self.reservoir.species == target_species:
                        self.inter[incident_species]['ndens'][it] = self.reservoir.get_density()
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
                        self._deposit_target_species(incident_species, target_species, jstarget)

        for incident_species in self.inter:
            self._incident_species = incident_species
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
                self._target_species = target_species
                ndens = self.inter[incident_species]['ndens'][it]
                target_fluidvel = self.inter[
                    incident_species]['target_fluidvel'][it]
                for js in incident_species.jslist:
                    i1 = ipg.ins[js] - 1 + top.it % self.stride
                    i2 = ipg.ins[js] + ipg.nps[js] - 1
                    self.xi = ipg.xp[i1:i2:self.stride]  # .copy()
                    self.yi = ipg.yp[i1:i2:self.stride]  # .copy()
                    self.zi = ipg.zp[i1:i2:self.stride]  # .copy()
                    ni = shape(self.xi)[0]
                    gaminvi = ipg.gaminv[i1:i2:self.stride]  # .copy()
                    self.uxi = ipg.uxp[i1:i2:self.stride]  # .copy()
                    self.uyi = ipg.uyp[i1:i2:self.stride]  # .copy()
                    self.uzi = ipg.uzp[i1:i2:self.stride]  # .copy()
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
                        setu_in_uzboosted_frame3d(ni, self.uxi, self.uyi, self.uzi, gaminvi,
                                                  -uzboost,
                                                  top.boost_gamma)
                    self.vxi = self.uxi * gaminvi
                    self.vyi = self.uyi * gaminvi
                    self.vzi = self.uzi * gaminvi
                    # --- get local target density
                    if ndens is None:
                        ndens = self.target_dens[target_species]['ndens']
                    if isinstance(ndens, (int, float)):
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
                            ri = sqrt(self.xi * self.xi + self.yi * self.yi)
                            dp = where((ri >= xmin) & (ri <= xmax) &
                                       (self.zi >= zmin) & (self.zi <= zmax), dp, 0.)
                        else:
                            dp = where((self.xi >= xmin) & (self.xi <= xmax) &
                                       (self.yi >= ymin) & (self.yi <= ymax) &
                                       (self.zi >= zmin) & (self.zi <= zmax), dp, 0.)
                    else:
                        dp = zeros(ni, 'd')
                        getgrid3d(ni, self.xi, self.yi, self.zi, dp,
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
                        getgrid3d(ni, self.xi, self.yi, self.zi, vxtf,
                                  self.nx, self.ny, self.nz, target_fluidvel[
                                      ..., 0],
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)
                        getgrid3d(ni, self.xi, self.yi, self.zi, vytf,
                                  self.nx, self.ny, self.nz, target_fluidvel[
                                      ..., 1],
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)
                        getgrid3d(ni, self.xi, self.yi, self.zi, vztf,
                                  self.nx, self.ny, self.nz, target_fluidvel[
                                      ..., 2],
                                  self.xmin, self.xmax, self.ymin, self.ymax, self.zmin, self.zmax,
                                  w3d.l2symtry, w3d.l4symtry)

                    # compute the relative velocity
                    # NOTE that at this point, the target species is assumed to have a negligible velocity.
                    # this needs to be modified if this approximation is not
                    # valid.
                    vxr = self.vxi - vxtf
                    vyr = self.vyi - vytf
                    vzr = self.vzi - vztf
                    vi = sqrt(vxr * vxr + vyr * vyr + vzr * vzr)
                    cross_section = self.getcross_section(
                        self.inter[incident_species]['cross_section'][it], vi)

                    # Number of collisions
                    #print("dp: {}".format(dp))
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
                    if _DEBUG:
                        print("starting ncoli:", ncoli)

                    # Select the particles that will collide
                    # I think this is just equivalent to np.where(ncoli>0) - cch
                    self.io = compress(ncoli > 0, arange(ni))
                    nnew = len(self.io)

                    if None in self.inter[incident_species]['emitted_energy0'][it]:
                        # When emitted_energy0 is not specified, use the velocity of
                        # the incident particles for the emitted particles.
                        uxnewsave = self.uxi
                        uynewsave = self.uyi
                        uznewsave = self.uzi

                    if self.inter[incident_species]['remove_incident'][it]:
                        # If projectile is modified, then need to delete it
                        put(ipg.gaminv, array(self.io) * self.stride + i1, 0.)

                    # The position of the incident particle is at or near the incident particle
                    xnew = self.xi
                    ynew = self.yi
                    znew = self.zi

                    incidentvelocities = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    originalvelocities = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    emissionangles = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    emittedvelocities = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}
                    recoilangles = {species: np.array([]) for species in self.inter[incident_species]['emitted_species'][it]}

                    # Loop until there are no more collision events that need handling
                    _counter = 0
                    while nnew > 0:

                        # The emitted particles positions, in some cases, are slightly
                        # offset from the incident
                        xnewp = xnew[self.io]
                        ynewp = ynew[self.io]
                        znewp = znew[self.io]
                        xnew = xnewp + (ranf(xnewp) - 0.5) * 1.e-10 * self.dx
                        ynew = ynewp + (ranf(ynewp) - 0.5) * 1.e-10 * self.dy
                        znew = znewp + (ranf(znewp) - 0.5) * 1.e-10 * self.dz
                        if top.wpid == 0:
                            w = 1.
                        else:
                            w = wi[self.io]

                        # The injdatapid value needs to be copied to the emitted particles
                        # so that they are handled properly in the region near the source.
                        if top.injdatapid > 0:
                            injdatapid = injdatapid[self.io]

                        # If the emitted energy was not specified, the emitted particle will be
                        # given the same velocity of the incident particle.
                        # TODO: Not sure if the None case for emitted_energy0 is properly handled
                        if None in self.inter[incident_species]['emitted_energy0'][it]:
                            uxnewsave = uxnewsave[self.io]
                            uynewsave = uynewsave[self.io]
                            uznewsave = uznewsave[self.io]
                        
                        n_emitted = len(self.inter[incident_species]['emitted_species'][it])
                        
                        # For each collision calculate product velocity components and modify incident particle
                        for ie, emitted_species in enumerate(self.inter[incident_species]['emitted_species'][it]):
                            
                            incident_ke = (incident_species.mass*clight**2)/jperev * (1. / ipg.gaminv[self.io * self.stride + i1] - 1.)

                            # If no emitted_energy0 then emission particle velocity set to incident particle velocity
                            if self.inter[incident_species]['emitted_energy0'][it][ie] is not None:
                                # Create new momenta for the emitted particles.
                                # Only macro particles participating in collisions (vi[io]) should be used in
                                #  the calculation. So nnew is no longer necessary if io is used to select indices
                                #  from vi, however, it is left as an argument to maintain backwards compatibility
                                emitted_energy0 = self.inter[incident_species]['emitted_energy0'][it][ie](self, vi=vi[self.io], nnew=nnew)
                                emitted_energy_sigma = self.inter[incident_species]['emitted_energy_sigma'][it][ie](self, vi=vi[self.io], nnew=nnew)
                                if emitted_energy_sigma is None:
                                    emitted_energy_sigma = 0
                                else:
                                    emitted_energy_sigma = np.random.normal(loc=0., scale=emitted_energy_sigma, size=nnew)

                                if self.l_verbose:
                                    print('incident:', incident_species.name, ',', 'emitted:', emitted_species.name)
                                    print("  emitted_energy0", emitted_energy0)
                                    print("  emitted_energy_sigma", emitted_energy_sigma)
                                    print()
                                #print("  emitted_energy0", emitted_energy0)
                                emitted_energy = emitted_energy0 + emitted_energy_sigma
                                # Initial calculation of emitted particle velocity components
                                gnew = 1. + emitted_energy * jperev / (emitted_species.mass * clight ** 2)
                                #if self.l_verbose:
                                #print("gamma new: ", gnew)
                                bnew = np.sqrt(1 - 1 / gnew ** 2)

                                ui = np.vstack((self.uxi[self.io], self.uyi[self.io], self.uzi[self.io])).T
                                incidentvelocities[emitted_species] = np.append(incidentvelocities[emitted_species], ui)
                                norm = np.linalg.norm(ui, axis=1)

                                uxnew = self.uxi[self.io] / norm * bnew * gnew * clight
                                uynew = self.uyi[self.io] / norm * bnew * gnew * clight
                                uznew = self.uzi[self.io] / norm * bnew * gnew * clight

                            else:
                                uxnew = uxnewsave
                                uynew = uynewsave
                                uznew = uznewsave

                            # Remove energy from incident particle if energy is not attributed to thermal motion
                            if (not self.inter[incident_species]['thermal'][it][ie]) and self.inter[incident_species]['conserve_energy'][it][ie]:
                                if self.l_verbose:
                                    print("scaling:", incident_species.name)
                                    print("ratios",
                                          emitted_energy / (incident_species.mass * warp.clight**2 / np.abs(incident_species.charge)))
                                scale = self._scale_primary_momenta(incident_species, ipg, emitted_energy, i1, i2, self.io)
                            elif self.inter[incident_species]['thermal'][it][ie]:
                                # Set emitted momenta from thermal motion but do not remove energy from incident
                                scale = 1.0
                                temperature = self.inter[incident_species]['temperature'][it][ie]
                                uxnew, uynew, uznew = self._generate_thermal_momentum(nnew, temperature, incident_species)
                            else:
                                scale = 1.0

                            self.uxi[self.io] *= scale
                            self.uyi[self.io] *= scale
                            self.uzi[self.io] *= scale

                            assert np.all(self.vzi != 0), "Not all components of vzi are non-zero"

                            v1 = np.vstack((self.vxi, self.vyi, self.vzi)).T[self.io]
                            v2 = v1.copy()
                            v2[:, 2] = 0
                            rotvec = np.cross(v1, v2)

                            # If emission angle spectrum is provided then sample and rotate emitted particles
                            # If not used then emission momentum vector is scaled from incident (above)
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
                                    rangles = np.array(self.sampleIncidentAngle(nnew=nnew, emitted_energy=emitted_energy, incident_energy=incident_ke, emitted_theta=emissionangles))
                                    if self.writeAngleDataDir and top.it % self.writeAnglePeriod == 0:
                                        recoilangles[emitted_species] = np.append(recoilangles[emitted_species], rangles)
                                    vin = np.vstack((self.vxi[self.io], self.vyi[self.io], self.vzi[self.io])).T
                                    self.vxi[self.io], self.vyi[self.io], self.vzi[self.io] = [l.flatten() for l in rotateVec(vec=vin, rotaxis=rotvec, theta=rangles)]

                            ginew = 1. / sqrt(1. + (uxnew**2 + uynew ** 2 + uznew**2) / clight**2)
                            #print("gamma inverse new: ", ginew)
                            # get velocity in boosted frame if using a boosted frame of reference
                            if top.boost_gamma > 1.:
                                setu_in_uzboosted_frame3d(shape(ginew)[0], uxnew, uynew, uznew, ginew,
                                                          uzboost,
                                                          top.boost_gamma)

                            if self.l_verbose:
                                print('add ', nnew, emitted_species.name, ' from by impact ionization:', incident_species.name, '+', ((target_species is None and 'background gas') or target_species.name))
                            if self.inter[incident_species]['remove_incident'][it] and (emitted_species.type is incident_species.type):
                                self.addpart(nnew, xnewp, ynewp, znewp, uxnew, uynew, uznew, ginew, epg, emitted_species.jslist[0],
                                             self.inter[incident_species]['emitted_tag'][it], injdatapid, w)
                            else:
                                self.addpart(nnew, xnew, ynew, znew, uxnew, uynew, uznew, ginew, epg, emitted_species.jslist[0],
                                             self.inter[incident_species]['emitted_tag'][it], injdatapid, w)
                                #print("Adding particle. nnew: {} and ndens: {}".format(nnew,ndens))

                                if self.reservoir and self.reservoir.species == emitted_species:
                                    # -1 because ionizatino removes density from neutral particle reservoir
                                    self.reservoir.deposit_density(xnew, ynew, znew, -1. * emitted_species.sw * np.ones_like(xnew))
    
                        ncoli = ncoli[self.io] - 1
                        if _DEBUG:
                            print("ncoli:", ncoli)
                        self.io = arange(nnew)[ncoli > 0]
                        if _DEBUG:
                            print("io:", self.io)
                        nnew = len(self.io)
                        if _DEBUG:
                            print("nnew new", nnew)
                        _counter += 1
                        if _DEBUG:
                            print(_counter)
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
                                if _DEBUG:
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
                    print('time ionization = ', time.clock() - t1, 's')

    def _deposit_target_species(self, incident_species, target_species, jstarget):
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
        # deposit density
        deposgrid3d(1, tpg.nps[jstarget], xt, yt, zt,
                    tpg.sw[jstarget] * self.invvol * fact * weights,
                    self.nx, self.ny, self.nz, self.target_dens[target_species]['ndens'],
                    self.ndensc,
                    self.xmin, self.xmax, self.ymin, self.ymax,
                    self.zmin, self.zmax)
        # computes target fluid velocity
        deposgrid3dvect(0, tpg.nps[jstarget], xt, yt, zt, vxt, vyt, vzt,
                        tpg.sw[jstarget] * self.invvol * fact * weights,
                        self.nx, self.ny, self.nz, self.target_dens[target_species]['target_fluidvel'],
                        self.ndensc,
                        self.xmin, self.xmax, self.ymin, self.ymax,
                        self.zmin, self.zmax)

    def _generate_thermal_momentum(self, nnew, temperature, incident_species):
        """
        Provide thermal velocity to emitted ions when the target is a gas reservoir
        Args:
            nnew (int): Number of new particles.
            temperature (float): Temperature of gas reservoir. Units of Kelvin.
            incident_species: warp.particles.species.Species instance for the incident species.

        Returns:
            px (ndarray), py (ndarray), pz (ndarray): Momentum vector components
                normalized by particle mass of emitted ions.
        """
        mass = self.inter[incident_species]['emitted_species'][0][0].mass  # kg
        var_xyz = boltzmann * temperature / mass
        var_vs = np.asarray([var_xyz, var_xyz, var_xyz])
        distr_mean = [0., 0., 0.]  # Each distribution has a native mean of 0.
        distr_cov = np.multiply(var_vs, np.identity(3))  # distributions are assumed to be independent

        velocity = np.random.multivariate_normal(distr_mean, distr_cov, nnew)
        gamma = 1. / np.sqrt(1. - (velocity / clight)**2)

        return (velocity * gamma)[:, 0], (velocity * gamma)[:, 1], (velocity * gamma)[:, 2]
