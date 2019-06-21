# sometimes the standard random is overwritten by numpy.random
# so random is renamed as std_random here

import numpy as np
from warp import *
import random as std_random
from scipy.constants import k, m_e

class ParticleReflector:
    """Class that handles particle reflection.

    To use:

      To define a particlereflector instance based on the installed scraper named "scraper"
      and the conductor named "collector" with the reflected species being "rc_electrons":
      >>> collector_reflector = particleconductor(scraper=scraper, conductor=collector,
                                                  spref=rc_electrons)

      The defined reflector must be installed, otherwise it will NOT be used:
      >>> installparticlereflector(collector_reflector)

      To define the specular reflection probability to be 0.2 (default 0),
      and the diffuse reflection probability to be 0.3 (default 0):
      >>> collector_reflector = particleconductor(scraper=scraper, conductor=collector,
                                                  spref=rc_electrons,
                                                  srefprob=0.2, drefprob=0.3)

      To only allow species "e1" and "e2" reflectable
      (default allows all defined species reflectable):
      >>> collector_reflector = particleconductor(scraper=scraper, conductor=collector,
                                                  spref=rc_electrons,
                                                  spinc=["e1", "e2"],
                                                  srefprob=0.2, drefprob=0.3)
    """
    def __init__(self, scraper=None, conductor=None, spref=None, spinc=None,
            srefprob=0., drefprob=0., refscheme="rand_angle", top=top):
        """Initialize a particle reflector.

        Args:
          scraper: Warp's scraper instance (must be specified).
          conductor: Warp's conductor instance to be treated as reflector (must be specified).
          spref: Reflected species (must be given)
          spinc: A list of incident species which are allowed to be reflected.
            All the species defined are allowed to be reflected if it is not specified.
          srefprob: specular reflection probability (default 0)
          drefprob: diffuse reflection probability (default 0)
          refscheme: Scheme of assigning velocity compoents  for reflected particles.
            Available options are "rand_angle" (default) and "uniform".
            "rand_angle" randomizes the angles in the spherical coordinate system
            for the reflected particles.
            "uniform" first assigns a random fraction (uniformly distributed between 0 and 1)
            of the particle's total kinetic energy to the normal component, and then
            randomly distributes the rest of the kinetic energy to the two tangent components.
          top: Warp's top module
        """
        if not isinstance(scraper, ParticleScraper):
            raise RuntimeError("scraper must be defined")
        else:
            if not scraper.lcollectlpdata:
                raise RuntimeError("lcollectlpdata flag must be True in scraper")
            if not scraper.lsaveintercept:
                raise RuntimeError("lsaveintercept flag must be True in scraper")
        
        if not conductor:
            raise RuntimeError("conductor must be defined")
        else:
            if not conductor in scraper.conductors:
                raise RuntimeError("conductor is not registered in scraper")
        
        if not spref:
            raise RuntimeError("reflected species (spref) must be defined")
        
        if srefprob < 0.:
            raise RuntimeError("srefprob cannot be negative")
        if drefprob < 0.:
            raise RuntimeError("drefprob cannot be negative")
        if srefprob + drefprob > 1.:
            raise RuntimeError("sum of srefprob and drefprob cannot exceed 1")
        
        self._top = top
        self._conductor = conductor
        self._spref = spref
        self._jsref = spref.js
        self._srefprob = srefprob
        self._drefprob = drefprob
        self._totrefprob = srefprob + drefprob
        self._srefprobratio = self._srefprob/(self._totrefprob+1e-14)

        if spinc is None:  # consider all sp as incident sp if spinc not defined
            self._jsinc = range(top.ns)
        else:
            try:
                self._jsinc = [s.js for s in spinc]
            except:
                self._jsinc = [spinc.js]
        
        if refscheme == "rand_angle":
            self._refscheme = 0
        elif refscheme == "uniform":
            self._refscheme = 1
        else:
            raise RuntimeError("Illegal refscheme value [{}]".format(refscheme))

        self._nsinc = len(self._jsinc)
        self._npslost_cumulative = np.zeros((self._nsinc,), dtype=np.uint64)
        self._nps_sref_res = np.zeros((self._nsinc,), dtype=np.float)
        self._nps_dref_res = np.zeros((self._nsinc,), dtype=np.float)

    def inject_particles(self):
        """Inject reflected particles.

        This function is passed to Warp through the interface "installparticlereflector",
        which wraps Warp's "installuserinjection" function.
        """
        
        for i, js in enumerate(self._jsinc):
            # skip if no particle lost in this step
            npslost = int(self._top.npslost[js] - self._npslost_cumulative[i])  # num of lost js particles for this step
            if npslost == 0: continue

            istart = int(self._npslost_cumulative[i] + self._top.inslost[js]-1)
            iend = istart + npslost

            plostindx = np.arange(istart, iend)

            # get a local reference
            # initial coordinates of particles at this step
            xpold = self._top.pidlost[plostindx, 0]
            ypold = self._top.pidlost[plostindx, 1]
            zpold = self._top.pidlost[plostindx, 2]

            # initial velocities of particles at this step
            uxpold = self._top.uxplost[plostindx]
            uypold = self._top.uyplost[plostindx]
            uzpold = self._top.uzplost[plostindx]

            # time step
            dt = self._top.dt
            # new position (inside conductor)
            xpinsi = uxpold*dt + xpold
            ypinsi = uypold*dt + ypold
            zpinsi = uzpold*dt + zpold

            # particles scraped by this conductor
            cinside = nint(self._conductor.isinside(xpinsi, ypinsi, zpinsi).isinside)
            cplostindx = compress(cinside, plostindx)
            cnpslost = len(cplostindx)

            # floating total num of particles to be reflected
            nps_totref = cnpslost*self._totrefprob
            
            # floating num of particles to be specularly and diffusively reflected
            nps_sref = nps_totref*self._srefprobratio
            nps_dref = nps_totref-nps_sref
            
            nps_sref_inj = int(nps_sref + self._nps_sref_res[i] + np.random.rand())
            nps_dref_inj = int(nps_dref + self._nps_dref_res[i] + np.random.rand())
            
            # if total num of particles to be injected exceeds total num of lost particles this step
            # then reduce num of particles to be injected 
            while nps_sref_inj + nps_dref_inj > cnpslost:
                if np.random.rand() < self._srefprobratio:
                    nps_sref_inj -= 1
                else:
                    nps_dref_inj -= 1
            
            # accumulate the residual number of particles
            # to correct the number of particles to be injected in the future
            self._nps_sref_res[i] += (nps_sref - nps_sref_inj)
            self._nps_dref_res[i] += (nps_dref - nps_dref_inj)
            
            if nps_sref_inj+nps_dref_inj > 0:

                # The current strategy is the fastest (at least for array of range(istart, iend))
                # alternatives purely based on numpy could be:
                # rand_indx = np.random.choice(np.arange(istart, iend), nps_sref_inj+nps_dref_inj, replace=False)
                # or
                # rand_indx = np.random.permutation(np.arange(istart, iend))[:nps_sref_inj+nps_dref_inj]
                # the two numpy strategies are essentially the same, but slower than the following implementation
                rand_indx = np.array(std_random.sample(range(cnpslost), nps_sref_inj+nps_dref_inj))

                if nps_sref_inj > 0:  # specular reflection
                    self._specular_reflection(cplostindx[rand_indx[:nps_sref_inj]])


                if nps_dref_inj > 0:  # diffuse reflection
                    self._diffuse_reflection(cplostindx[rand_indx[nps_sref_inj:]])

                
#                 self._conductor.emitparticles_data.append(array([top.time,
#                                                                  totemit,
#                                                                  top.dt,
#                                                                  self.inter[js]['emitted_species'][ics][ie].jslist[0],
#                                                                  totabsorb]))
                self._conductor.emitparticles_data.append(array([self._top.time,
                                                                 self._top.dt,
                                                                 js,
                                                                 nps_sref_inj*self._top.pgroup.sq[self._jsref]*self._top.pgroup.sw[self._jsref],
                                                                 nps_dref_inj*self._top.pgroup.sq[self._jsref]*self._top.pgroup.sw[self._jsref]]))
                
            self._npslost_cumulative[i] = self._top.npslost[js]
        
        return
            
    def _specular_reflection(self, selected_lpid):
        """Perform specular reflection.

        Args:
          selected_lpid: A list of particle id selected to be reflected specularly.
        """

        pos_intsec = self._get_intersection(selected_lpid)

        nvec = self._get_surface_normal(selected_lpid)

        vel_inc = self._get_incident_velocity(selected_lpid)

        # velocities of reflected particles
        vel_ref = self._vel_specular_reflection(nvec, vel_inc)

        t_left = self._get_time_left(selected_lpid)

        # coordinates of reflected particles
        pos_ref = self._pos_reflection(t_left, pos_intsec, vel_ref)

        # add the new reflected particles to "reflected species"
        self._spref.addparticles(x=pos_ref[:,0], y=pos_ref[:,1], z=pos_ref[:,2],
                                 vx=vel_ref[:,0], vy=vel_ref[:,1], vz=vel_ref[:,2])

        return

    def _diffuse_reflection(self, selected_lpid):
        """Perform diffuse reflection.

        Args:
          selected_lpid: A list of particle id selected to be reflected diffusively.
        """

        pos_intsec = self._get_intersection(selected_lpid)

        nvec = self._get_surface_normal(selected_lpid)

        vel_inc = self._get_incident_velocity(selected_lpid)

        # velocities of reflected particles
        vel_ref = self._vel_diffuse_reflection(nvec, vel_inc)

        t_left = self._get_time_left(selected_lpid)

        # coordinates of reflected particles
        pos_ref = self._pos_reflection(t_left, pos_intsec, vel_ref)

        # add the new reflected particles to "reflected species"
        self._spref.addparticles(x=pos_ref[:,0], y=pos_ref[:,1], z=pos_ref[:,2],
                                 vx=vel_ref[:,0], vy=vel_ref[:,1], vz=vel_ref[:,2])

        return

    def _pos_reflection(self, t_left, pos_intsec, v_re):
        """Calculate reflected particles' new positions

        Args:
          t_left: A list of time left for particles after they hit the reflector.
          pos_intsec: A list of position coordinates where particles hit the reflector.
          v_re: A list of reflected particles' new velocities.

        Return:
          pos_re: Reflected particles' new positions.
        """

        pos_re = pos_intsec + v_re*t_left

        return pos_re

    def _vel_specular_reflection(self, nvec, v_in):
        """Calculate new velocities for specularly reflected particles.

        Args:
          nvec: Normal vectors for the reflector surfaces at which particles hit.
          v_in: Velocities of incident particles.

        Return:
          v_re: New velocities of reflected particles.
        """

        size = v_in.shape[0]
        v_in_n = np.zeros_like(v_in)
        for i in range(size):
            dotprod = np.dot(v_in[i,:], nvec[i,:])
            v_in_n[i] = dotprod*nvec[i,:]
        v_in_t = v_in - v_in_n
        v_re_n = -v_in_n
        v_re_t = v_in_t
        v_re = v_re_n + v_re_t
        return v_re

    def _vel_diffuse_reflection(self, nvec, v_in):
        """Calculate new velocities for diffusively reflected particles.

        Args:
          nvec: Normal vectors for the reflector surfaces at which particles hit.
          v_in: Velocities of incident particles.

        Return:
          v_re: New velocities of reflected particles.
        """

        size = v_in.shape[0]
        v_re = np.zeros_like(v_in)
        
        ###############################################
        # redistribute post-kinetic energy            #
        ###############################################
        if self._refscheme == 0:
            # S0: randomize velocity spherical coordinate

            v_mag = np.sqrt(v_in[:,0]**2 + v_in[:,1]**2 + v_in[:,2]**2)

            # spherical coordinates theta [0~pi/2], phi [0~2*pi]
            theta = np.random.rand(size)*0.5*np.pi
            phi = np.random.rand(size)*2.0*np.pi

            # velocity aligned with the conductor coordinate system
            vn  = v_mag*np.cos(theta)       # normal component
            vt1 = v_mag*np.sin(theta)
            vt2 = vt1*np.sin(phi)           # 1st tangent component
            vt1[:] *= np.cos(phi)           # 2nd tangent component
        elif self._refscheme == 1:
            # S1: randomize En with a uniform distribution

            vsq = v_in[:,0]**2 + v_in[:,1]**2 + v_in[:,2]**2
            vn = vsq*np.random.rand(size)
            vt1 = vsq - vn
            vt2 = vt1*np.random.rand(size)
            vt1 = vt1 - vt2
            vn = np.sqrt(vn)
            vt1 = np.sqrt(vt1)
            vt2 = np.sqrt(vt2)
        
        ###############################################
        # convert to Cartesian coordinate system      #
        ###############################################
        t1, t2 = self._get_tangent_from_normal(nvec)

        v_re[:, 0] = vn*nvec[:, 0] + vt1*t1[:, 0] + vt2*t2[:, 0]
        v_re[:, 1] = vn*nvec[:, 1] + vt1*t1[:, 1] + vt2*t2[:, 1]
        v_re[:, 2] = vn*nvec[:, 2] + vt1*t1[:, 2] + vt2*t2[:, 2]
        
        return v_re

    def _get_intersection(self, selected_lpid):
        # particles' intersections with the conductor
        xx = self._top.xplost[selected_lpid]
        yy = self._top.yplost[selected_lpid]
        zz = self._top.zplost[selected_lpid]

        return np.array([xx, yy, zz]).transpose()

    def _get_surface_normal(self, selected_lpid):

        theta = self._top.pidlost[selected_lpid, -3]
        phi = self._top.pidlost[selected_lpid, -2]

        # convert spherical coordinate angles to normal vector
        return self._normal_spherical_to_cartesian(theta, phi)

    def _get_incident_velocity(self, selected_lpid):
        vx = self._top.uxplost[selected_lpid]
        vy = self._top.uyplost[selected_lpid]
        vz = self._top.uzplost[selected_lpid]

        return np.array([vx, vy, vz]).transpose()

    def _get_time_left(self, selected_lpid):
        t_left = self._top.time - self._top.pidlost[selected_lpid, -4]

        return t_left.reshape(len(t_left), 1)

    def _normal_spherical_to_cartesian(self, theta, phi):
        """Convert the normal vector in spherical coordinate system to Cartesian.

        The surface normal obtained from Warp is in spherical coordinate system.
        This is converted to Cartesian coordinate system in order to easily
        compute the reflected particles' new velocities and positions.
        """

        nvec = np.zeros((len(theta), 3))
        nvec[:, 0] = np.sin(theta)*np.cos(phi)
        nvec[:, 1] = np.sin(theta)*np.sin(phi)
        nvec[:, 2] = np.cos(theta)
        return nvec

    def _get_tangent_from_normal(self, nvec):
        """Compute two tangent vectors based on the surface normal.

        This is needed when converting the velocties and positions
        of particles in the surface coordinate system back to
        the lab coordinate system.
        """
        a = np.array([0., 0., 1.])
        b = np.array([0., 1., 0.])

        t1v = np.zeros_like(nvec)
        t2v = np.zeros_like(nvec)

        for i in range(nvec.shape[0]):
            n = nvec[i,:]
            c1 = np.cross(n, a)
            c2 = np.cross(n, b)
            norm_c1 = np.linalg.norm(c1)
            norm_c2 = np.linalg.norm(c2)
            if (norm_c1 > norm_c2):
                t1 = c1/norm_c1
            else:
                t1 = c2/norm_c2
            t2 = np.cross(n, t1)    # second tangent
            
            t1v[i,:] = t1
            t2v[i,:] = t2

        return t1v, t2v


# end of class ParticleReflector


def installparticlereflector(pr):
    """Install particle reflector.

    Args:
      pr: Particle reflector instance already defined.
    """

    if not isinstance(pr, ParticleReflector):
        raise RuntimeError("Illegal ParticleReflector instance")
    installuserinjection(pr.inject_particles)


def analyze_collected_charge(top, solver):
    """Analyze charges collected by all conductors.

    Args:
      top: Warp's top module.
      solver: Warp's field solver.

    Return:
      collected_charge: A list of collected particle info for all conductors and all species.
                        collected_charge[conductor_id][species_id]
    """

    nspecies = top.ns
    cond_ids = []
    cond_objs = []
    collected_charge = {}
    for cond in solver.conductordatalist:
        cond_objs.append(cond[0])
        cond_ids.append(cond[0].condid)

    for i, ids in enumerate(cond_ids):
        collected_charge[ids] = [[] for _ in range(nspecies)]
        for js in range(nspecies):
            jsid = top.pgroup.sid[js]
            indx = np.ndarray.astype(cond_objs[i].lostparticles_data[:, 3] + 1e-6, 'int') == jsid
            collected_charge[ids][js] = np.copy(cond_objs[i].lostparticles_data[indx, 0:4])

    return collected_charge


def analyze_reflected_charge(top, reflectors, comm_world=None):
    """Analyze charges due to reflection.

    Args:
      top: Warp's top module.
      reflectors: A list of particle reflectors.
      comm_wold: MPI communicator, must be passed in if running in parallel.

    Return:
      reflected_charge: A list of charges reflected particle info
                        for the corresponding reflector and species.
                        reflected_charge[reflector's_cond_id]
    """

    cond_ids = []
    cond_objs = []
    reflected_charge = {}
    for reflector in reflectors:
        cond_objs.append(reflector._conductor)
        cond_ids.append(reflector._conductor.condid)
    for i, ids in enumerate(cond_ids):
        reflected_charge[ids] = np.copy(cond_objs[i].emitparticles_data[:, 0:5])
    if comm_world:
        all_reflected_charge = {}
        for ids in cond_ids:
            all_reflected_charge[ids] = comm_world.gather(reflected_charge[ids], root=0)
            if comm_world.rank == 0:
                all_reflected_charge[ids] = np.vstack(all_reflected_charge[ids])
        return all_reflected_charge
    else:
        return reflected_charge
