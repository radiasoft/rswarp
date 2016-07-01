#!/usr/bin/env python
"""
Ionization class derived from Warp's Ionization, with some improvements.

6/30/2016 - Added support for emitted_energy0 and emitted_energy_sigma to be passed in as functions.
"""
from warp import *
from warp.particles import ionization
import numpy as np
import time
import types

__all__ = ['Ionization']


def tryFunctionalForm(f,*args,**kwargs):
    try:
        res = f(*args,**kwargs)
    except TypeError:
        res = f
    return res


class Ionization(ionization.Ionization):
    def generate(self,dt=None):
        if dt is None:dt=top.dt
        if self.l_timing:t1 = time.clock()
        for target_species in self.target_dens:
          self.target_dens[target_species]['ndens_updated']=0
        for incident_species in self.inter:
          npinc = 0
          ispushed=0
          ipg=self.inter[incident_species]['incident_pgroup']
          tpg=self.inter[incident_species]['target_pgroup']
          epg=self.inter[incident_species]['emitted_pgroup']
          for js in incident_species.jslist:
            npinc+=ipg.nps[js]
            if ipg.ldts[js]:ispushed=1
          if npinc==0 or not ispushed:continue
          for it,target_species in enumerate(self.inter[incident_species]['target_species']):
            ndens=self.inter[incident_species]['ndens'][it]
            target_fluidvel=self.inter[incident_species]['target_fluidvel'][it]
            if ndens is not None:
              continue
            else:
              if self.target_dens[target_species]['ndens_updated']:
                continue
              else:
                self.target_dens[target_species]['ndens_updated']=1
              ndens = self.target_dens[target_species]['ndens']
              target_fluidvel=self.target_dens[target_species]['target_fluidvel']
              nptarget=0
              for jstarget in target_species.jslist:
                nptarget+=tpg.nps[jstarget]
              if nptarget==0:continue
              self.ndensc[...]=0.
              ndens[...]=0.

              for jstarget in target_species.jslist:
                  self.depositTargetSpecies(jstarget)
    #          if w3d.l2symtry or w3d.l4symtry:self.ndens[:,0,:]*=2.
    #          if w3d.l4symtry:self.ndens[0,:,:]*=2.

        for incident_species in self.inter:
          npinc = 0
          ispushed=0
          ipg=self.inter[incident_species]['incident_pgroup']
          tpg=self.inter[incident_species]['target_pgroup']
          epg=self.inter[incident_species]['emitted_pgroup']
          for js in incident_species.jslist:
            npinc+=ipg.nps[js]
            if ipg.ldts[js]:ispushed=1
          if npinc==0 or not ispushed:continue
          for it,target_species in enumerate(self.inter[incident_species]['target_species']):
            ndens=self.inter[incident_species]['ndens'][it]
            target_fluidvel=self.inter[incident_species]['target_fluidvel'][it]
    #        ndens=1.e23
            for js in incident_species.jslist:
              i1 = ipg.ins[js] - 1 + top.it%self.stride
              i2 = ipg.ins[js] + ipg.nps[js] - 1
              xi=ipg.xp[i1:i2:self.stride]#.copy()
              yi=ipg.yp[i1:i2:self.stride]#.copy()
              zi=ipg.zp[i1:i2:self.stride]#.copy()
              ni = shape(xi)[0]
              gaminvi=ipg.gaminv[i1:i2:self.stride]#.copy()
              uxi=ipg.uxp[i1:i2:self.stride]#.copy()
              uyi=ipg.uyp[i1:i2:self.stride]#.copy()
              uzi=ipg.uzp[i1:i2:self.stride]#.copy()
              if top.wpid > 0:
                # --- Save the wpid of the incident particles so that it can be
                # --- passed to the emitted particles.
                wi = ipg.pid[i1:i2:self.stride,top.wpid-1]
              else:
                wi = 1.
              if top.injdatapid > 0:
                # --- Save the injdatapid of the incident particles so that it can be
                # --- passed to the emitted particles.
                injdatapid = ipg.pid[i1:i2:self.stride,top.injdatapid-1]
              else:
                injdatapid = None
              # --- get velocity in lab frame if using a boosted frame of reference
              if top.boost_gamma>1.:
                uzboost = clight*sqrt(top.boost_gamma**2-1.)
                setu_in_uzboosted_frame3d(ni,uxi,uyi,uzi,gaminvi,
                                          -uzboost,
                                          top.boost_gamma)
              vxi=uxi*gaminvi
              vyi=uyi*gaminvi
              vzi=uzi*gaminvi
              # --- get local target density
              if ndens is None:
                ndens = self.target_dens[target_species]['ndens']
              if isinstance(ndens,(types.IntType,float)):
                dp=ones(ni,'d')*ndens
                if target_fluidvel is None:
                  xmin=self.xmin
                  xmax=self.xmax
                  ymin=self.ymin
                  ymax=self.ymax
                  zmin=self.zmin
                  zmax=self.zmax
                else:
                  vxtf = target_fluidvel[0]
                  vytf = target_fluidvel[1]
                  vztf = target_fluidvel[2]
                  xmin=self.xmin+vxtf*top.time
                  xmax=self.xmax+vxtf*top.time
                  ymin=self.ymin+vytf*top.time
                  ymax=self.ymax+vytf*top.time
                  zmin=self.zmin+vztf*top.time
                  zmax=self.zmax+vztf*top.time
                if w3d.solvergeom==w3d.RZgeom:
                  ri=sqrt(xi*xi+yi*yi)
                  dp=where((ri>=xmin) & (ri<=xmax) & \
                           (zi>=zmin) & (zi<=zmax),dp,0.)
                else:
                  dp=where((xi>=xmin) & (xi<=xmax) & \
                           (yi>=ymin) & (yi<=ymax) & \
                           (zi>=zmin) & (zi<=zmax),dp,0.)
              else:
                dp=zeros(ni,'d')
                getgrid3d(ni,xi,yi,zi,dp,
                            self.nx,self.ny,self.nz,ndens, \
                            self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,
                            w3d.l2symtry,w3d.l4symtry)
              # --- get local target fluid velocity
              if target_fluidvel is None:
                if target_species is None:
                  target_fluidvel = [0.,0.,0.]
                else:
                  target_fluidvel = self.target_dens[target_species]['target_fluidvel']
              if isinstance(target_fluidvel,list):
                vxtf = target_fluidvel[0]
                vytf = target_fluidvel[1]
                vztf = target_fluidvel[2]
              else:
                vxtf=zeros(ni,'d')
                vytf=zeros(ni,'d')
                vztf=zeros(ni,'d')
                getgrid3d(ni,xi,yi,zi,vxtf,
                            self.nx,self.ny,self.nz,target_fluidvel[...,0], \
                            self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,
                            w3d.l2symtry,w3d.l4symtry)
                getgrid3d(ni,xi,yi,zi,vytf,
                            self.nx,self.ny,self.nz,target_fluidvel[...,1], \
                            self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,
                            w3d.l2symtry,w3d.l4symtry)
                getgrid3d(ni,xi,yi,zi,vztf,
                            self.nx,self.ny,self.nz,target_fluidvel[...,2], \
                            self.xmin,self.xmax,self.ymin,self.ymax,self.zmin,self.zmax,
                            w3d.l2symtry,w3d.l4symtry)

              # compute the relative velocity
              # NOTE that at this point, the target species is assumed to have a negligible velocity.
              # this needs to be modified if this approximation is not valid.
              vxr = vxi-vxtf
              vyr = vyi-vytf
              vzr = vzi-vztf
              vi=sqrt(vxr*vxr+vyr*vyr+vzr*vzr)
              cross_section = self.getcross_section(self.inter[incident_species]['cross_section'][it],vi)

              # probability
              ncol = dp*cross_section*vi*dt*ipg.ndts[js]*self.stride
              if top.boost_gamma>1.:ncol*=top.gammabar_lab/top.gammabar

              # --- If the incident species is being removed, then only one collision event can happen.
              # --- Otherwise, a single particle may collide multiple times in a time step if ncol > 1.
              if self.inter[incident_species]['remove_incident'][it]:
                # --- Note that ncol is being set to slightly less than one. In the code below, adding ranf
                # --- to it will guarantee that it will end up as one. (Is this needed?)
                ncol = where(ncol>=1.,1.-1.e-10,ncol)

              # --- Get a count of the number of collisions for each particle. A random number is added to
              # --- ncol so that a fractional value has chance to result in a collision.
              ncoli=aint(ncol + ranf(ncol))

              # --- Select the particles that will collide
              io=compress(ncoli>0,arange(ni))
              nnew = len(io)

              if None in self.inter[incident_species]['emitted_energy0'][it]:
                # --- When emitted_energy0 is not specified, use the velocity of
                # --- the incident particles for the emitted particles.
                uxnewsave = uxi
                uynewsave = uyi
                uznewsave = uzi

              if self.inter[incident_species]['remove_incident'][it]:
                # --- if projectile is modified, then need to delete it
                put(ipg.gaminv,array(io)*self.stride+i1,0.)

              # --- The position of the incident particle is at or near the incident particle
              xnew = xi
              ynew = yi
              znew = zi

              # --- Loop until there are no more collision events that need handling
              while(nnew>0):

                # --- The emitted particles positions, in some cases, are slightly
                # --- offset from the incident
                xnewp = xnew[io]
                ynewp = ynew[io]
                znewp = znew[io]
                xnew = xnewp+(ranf(xnewp)-0.5)*1.e-10*self.dx
                ynew = ynewp+(ranf(ynewp)-0.5)*1.e-10*self.dy
                znew = znewp+(ranf(znewp)-0.5)*1.e-10*self.dz
                if top.wpid == 0:
                  w = 1.
                else:
                  w = wi[io]

                # --- The injdatapid value needs to be copied to the emitted particles
                # --- so that they are handled properly in the region near the source.
                if top.injdatapid > 0: injdatapid = injdatapid[io]

                # --- If the emitted energy was not specified, the emitted particle will be
                # --- given the same velocity of the incident particle.
                if None in self.inter[incident_species]['emitted_energy0'][it]:
                  uxnewsave = uxnewsave[io]
                  uynewsave = uynewsave[io]
                  uznewsave = uznewsave[io]

                for ie,emitted_species in enumerate(self.inter[incident_species]['emitted_species'][it]):

                  if self.inter[incident_species]['emitted_energy0'][it][ie] is not None:
                    # --- Create new velocities for the emitted particles.
                    ek0ionel = tryFunctionalForm(self.inter[incident_species]['emitted_energy0'][it][ie],vi)
                    esigionel = tryFunctionalForm(self.inter[incident_species]['emitted_energy_sigma'][it][ie],vi)
                    if esigionel==0.:
                      ek = zeros(nnew)
                    else:
                      ek = SpRandom(0.,esigionel,nnew) #kinetic energy
                    ek=abs(ek+ek0ionel) #kinetic energy
                    fact = jperev/(emass*clight**2)
                    gamma=ek*fact+1.
                    u=clight*sqrt(ek*fact*(gamma+1.))
                    # velocity direction: random in (x-y) plane plus small longitudinal component:
                    phi=2.*pi*ranf(u)
                    vx=cos(phi);
                    vy=sin(phi);
                    vz=0.01*ranf(u)
                    # convert into a unit vector:
                    vu=sqrt(vx**2+vy**2+vz**2)
                    # renormalize:
                    vx/=vu; vy/=vu; vz/=vu
                    # find components of v*gamma:
                    uxnew=u*vx
                    uynew=u*vy
                    uznew=u*vz
                  else:
                    uxnew = uxnewsave
                    uynew = uynewsave
                    uznew = uznewsave

                  ginew = 1./sqrt(1.+(uxnew**2+uynew**2+uznew**2)/clight**2)
                  # --- get velocity in boosted frame if using a boosted frame of reference
                  if top.boost_gamma>1.:
                    setu_in_uzboosted_frame3d(shape(ginew)[0],uxnew,uynew,uznew,ginew,
                                              uzboost,
                                              top.boost_gamma)

                  if self.l_verbose:print 'add ',nnew, emitted_species.name,' from by impact ionization:',incident_species.name,'+',((target_species is None and 'background gas') or target_species.name)
                  if self.inter[incident_species]['remove_incident'][it] and (emitted_species.type is incident_species.type):
                    self.addpart(nnew,xnewp,ynewp,znewp,uxnew,uynew,uznew,ginew,epg,emitted_species.jslist[0],
                                 self.inter[incident_species]['emitted_tag'][it],injdatapid,w)
                  else:
                    self.addpart(nnew,xnew,ynew,znew,uxnew,uynew,uznew,ginew,epg,emitted_species.jslist[0],
                                 self.inter[incident_species]['emitted_tag'][it],injdatapid,w)
                ncoli = ncoli[io] - 1
                io = arange(nnew)[ncoli>0]
                nnew = len(io)

        # make sure that all particles are added and cleared
        for pg in self.x:
          for js in self.x[pg]:
            self.flushpart(pg,js)

        for incident_species in self.inter:
          for js in incident_species.jslist:
            processlostpart(top.pgroup,js+1,top.clearlostpart,top.time,top.zbeam)

            if self.l_timing:print 'time ionization = ',time.clock()-t1,'s'


    def depositTargetSpecies(self, jstarget):
        """ Depositing target species to the grid """
        tpg=self.inter[incident_species]['target_pgroup']
        i1 = tpg.ins[jstarget] - 1 # ins is index of first member of species
        i2 = tpg.ins[jstarget] + tpg.nps[jstarget] - 1
        xt=tpg.xp[i1:i2]
        yt=tpg.yp[i1:i2]
        zt=tpg.zp[i1:i2]
        git=tpg.gaminv[i1:i2]
        vxt=tpg.uxp[i1:i2]*git
        vyt=tpg.uyp[i1:i2]*git
        vzt=tpg.uzp[i1:i2]*git
        fact=1.
        if w3d.l4symtry:
          xt=abs(xt)
          fact=0.25
        elif w3d.l2symtry:
          fact=0.5
        if w3d.l2symtry or w3d.l4symtry:yt=abs(yt)
        if top.wpid==0:
          weights=ones(tpg.nps[jstarget],'d')
        else:
          weights=tpg.pid[i1:i2,top.wpid-1]
        # --- deposit density
        deposgrid3d(1,tpg.nps[jstarget],xt,yt,zt,
                    tpg.sw[jstarget]*self.invvol*fact*weights,
                    self.nx,self.ny,self.nz,ndens,self.ndensc,
                    self.xmin,self.xmax,self.ymin,self.ymax,
                    self.zmin,self.zmax)
        # --- computes target fluid velocity
        deposgrid3dvect(0,tpg.nps[jstarget],xt,yt,zt,vxt,vyt,vzt,
                    tpg.sw[jstarget]*self.invvol*fact*weights,
                    self.nx,self.ny,self.nz,target_fluidvel,self.ndensc,
                    self.xmin,self.xmax,self.ymin,self.ymax,
                    self.zmin,self.zmax)
