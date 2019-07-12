from warp.particles import particlescraper
from warp import *

class ParticleScraperGrid(particlescraper.ParticleScraper):
    """Override scrape to allow for probability of particle being scraped and directional filtering"""
    def scrape(self, js):
        """Apply scraping to species js. It is better to call scrapeall. This will normally be called automatically."""
        # --- If there are no particles in this species, that nothing needs to be done

        if top.pgroup.nps[js] == 0: return

        if self.surfacespecies != None:
            if self.surfacespecies.jslist[0] == js:
                return  # Do not scrape any surface species

        # --- Get mesh information into local variables
        dx, dy, dz, nx, ny, nz, ix, iy, iz = self.grid.getmeshsize(self.mglevel)
        xmin = self.grid.xmmin + ix * dx
        xmax = self.grid.xmmin + (ix + nx) * dx
        ymin = self.grid.ymmin + iy * dy
        ymax = self.grid.ymmin + (iy + ny) * dy
        zmin = self.grid.zmmin + iz * dz + top.zbeam
        zmax = self.grid.zmmin + (iz + nz) * dz + top.zbeam

        # --- The ixa etc are the location of the x=0 plane. This is needed
        # --- since in certain cases, the size of the collapsed dimension
        # --- can be greater than one and the 0 plane needs to be found, it
        # --- is the plane where the data is stored. This is true for example
        # --- with the RZ EM solver.
        # --- The check of nx > 0 etc is done since in certain cases the xmin
        # --- can be nonzero when nx is zero, giving an erroneous value for ixa.
        # --- This is the case when the quasistatic solver is being used.
        ixa = iya = iza = 0
        if nx > 0: ixa = nint(-xmin / dx)
        if ny > 0: iya = nint(-ymin / dy)
        if nz > 0: iza = nint(-zmin / dz)
        isinside = self.grid.isinside

        # --- Get handy references to the particles in the species
        i1 = top.pgroup.ins[js] - 1
        i2 = top.pgroup.ins[js] + top.pgroup.nps[js] - 1
        xx = top.pgroup.xp[i1:i2]
        yy = top.pgroup.yp[i1:i2]
        zz = top.pgroup.zp[i1:i2]
        pp = zeros(top.pgroup.nps[js], 'd')
        # if js==1:print js,i1,i2,top.pgroup.zp[i1:i2],top.zbeam

        # --- Find which particles are close to a conductor. This
        # --- interpolates from the isinside grid. The results are
        # --- put into the array pp.
        if w3d.solvergeom in [w3d.XYZgeom]:
            getgrid3d(top.pgroup.nps[js], xx, yy, zz, pp,
                      nx, ny, nz, isinside, xmin, xmax, ymin, ymax, zmin, zmax,
                      w3d.l2symtry, w3d.l4symtry)
        elif w3d.solvergeom == w3d.RZgeom:
            # --- Note that for RZ, the radius is calculated for this, but
            # --- the original particle position is used below.
            rr = sqrt(xx ** 2 + yy ** 2)
            getgrid2d(top.pgroup.nps[js], rr, zz, pp, nx, nz, isinside[:, iya, :],
                      xmin, xmax, zmin, zmax)
        elif w3d.solvergeom == w3d.XZgeom:
            xsym, ysym = self.applysymmetry(xx, 0)
            getgrid2d(top.pgroup.nps[js], xsym, zz, pp, nx, nz, isinside[:, iya, :],
                      xmin, xmax, zmin, zmax)
        elif w3d.solvergeom == w3d.XYgeom:
            xsym, ysym = self.applysymmetry(xx, yy)
            getgrid2d(top.pgroup.nps[js], xsym, ysym, pp, nx, ny, isinside[:, :, iza],
                      xmin, xmax, ymin, ymax)
        elif w3d.solvergeom == w3d.Rgeom:
            # --- Note that for R, the radius is calculated for this, but
            # --- the original particle position is used below.
            rr = sqrt(xx ** 2 + yy ** 2)
            getgrid1d(top.pgroup.nps[js], rr, pp, nx, isinside[:, iya, iza],
                      xmin, xmax)
        elif w3d.solvergeom == w3d.Ygeom:
            xsym, ysym = self.applysymmetry(0, yy)
            getgrid1d(top.pgroup.nps[js], ysym, pp, ny, isinside[ixa, :, iza],
                      ymin, ymax)
        elif w3d.solvergeom == w3d.Zgeom:
            getgrid1d(top.pgroup.nps[js], zz, pp, nz, isinside[ixa, iya, :],
                      ymin, ymax)
        else:
            raise Exception("The particle scraping only works for XYZ, XY, RZ, R, Y and Z geometry")

        # --- Get indices for all of the particles which are close to a
        # --- conductor. If there are none, then immediately return.
        # --- Note, of course, that close may mean inside.
        iclose = compress(pp > 0., arange(i1, i2))
        if len(iclose) == 0: return

        # --- Get the positions of particles which are close to a conductor.
        xx = take(xx, iclose - i1)
        yy = take(yy, iclose - i1)
        zz = take(zz, iclose - i1)

        # --- The 'g' lists give the locations of the corners of the grid cell
        # --- relative to the grid location of the particles close to a
        # --- conductor. Also, get those grid locations.
        if w3d.solvergeom in [w3d.XYZgeom]:
            nd = 3
            gdx = [0., dx, 0., dx, 0., dx, 0., dx]
            gdy = [0., 0., dy, dy, 0., 0., dy, dy]
            gdz = [0., 0., 0., 0., dz, dz, dz, dz]
            xg = xmin + aint(abs(xx - xmin) / dx) * dx
            yg = ymin + aint(abs(yy - ymin) / dy) * dy
            zg = zmin + aint(abs(zz - zmin) / dz) * dz
        elif w3d.solvergeom in [w3d.RZgeom]:
            nd = 2
            gdx = [0., dx, 0., dx]
            gdz = [0., 0., dz, dz]
            # --- Like above, the radius is calculated in the temporary, but the
            # --- original particle position is used below.
            # --- These two lines calculating rr give the same result, but the second
            # --- is probably faster
            # rr = sqrt(xx**2 + yy**2)
            rr = take(rr, iclose - i1)
            xg = xmin + aint(abs(rr - xmin) / dx) * dx
            zg = zmin + aint(abs(zz - zmin) / dz) * dz
        elif w3d.solvergeom in [w3d.XZgeom]:
            nd = 2
            gdx = [0., dx, 0., dx]
            gdz = [0., 0., dz, dz]
            xg = xmin + aint(abs(xx - xmin) / dx) * dx
            zg = zmin + aint(abs(zz - zmin) / dz) * dz
        elif w3d.solvergeom == w3d.XYgeom:
            nd = 2
            gdx = [0., dx, 0., dx]
            gdy = [0., 0., dy, dy]
            xg = xmin + aint(abs(xx - xmin) / dx) * dx
            yg = ymin + aint(abs(yy - ymin) / dy) * dy
        elif w3d.solvergeom in [w3d.Rgeom]:
            nd = 1
            gdx = [0., dx]
            # --- Like above, the radius is calculated in the temporary, but the
            # --- original particle position is used below.
            # --- These two lines calculating rr give the same result, but the second
            # --- is probably faster
            # rr = sqrt(xx**2 + yy**2)
            rr = take(rr, iclose - i1)
            xg = xmin + aint(abs(rr - xmin) / dx) * dx
        elif w3d.solvergeom == w3d.Ygeom:
            nd = 1
            gdy = [0., dy]
            yg = ymin + aint(abs(yy - ymin) / dy) * dy
        elif w3d.solvergeom == w3d.Zgeom:
            nd = 1
            gdz = [0., dz]
            zg = zmin + aint(abs(zz - zmin) / dz) * dz

        nn = len(iclose)
        pp = zeros(nn, 'd')

        # --- Loop over the corners of the grid cell
        for i in range(2 ** nd):

            # --- Get id of the conductor that the particles are near
            # --- See comments in updateconductors regarding reducedisinside
            # --- An optimization trick is to shift the grid rather than
            # --- the particles, avoiding adding scalars to arrays.
            if w3d.solvergeom in [w3d.XYZgeom]:
                getgridngp3d(nn, xg, yg, zg, pp, nx, ny, nz, self.reducedisinside,
                             xmin - gdx[i], xmax - gdx[i], ymin - gdy[i], ymax - gdy[i],
                             zmin - gdz[i], zmax - gdz[i], 0., w3d.l2symtry, w3d.l4symtry)
            elif w3d.solvergeom in [w3d.XZgeom, w3d.RZgeom]:
                xgsym, ygsym = self.applysymmetry(xg, 0)
                getgridngp2d(nn, xgsym, zg, pp, nx, nz, self.reducedisinside[:, iya, :],
                             xmin - gdx[i], xmax - gdx[i], zmin - gdz[i], zmax - gdz[i])
            elif w3d.solvergeom == w3d.XYgeom:
                xgsym, ygsym = self.applysymmetry(xg, yg)
                getgridngp2d(nn, xgsym, ygsym, pp, nx, ny, self.reducedisinside[:, :, iza],
                             xmin - gdx[i], xmax - gdx[i], ymin - gdy[i], ymax - gdy[i])
            elif w3d.solvergeom == w3d.Rgeom:
                xgsym, ygsym = self.applysymmetry(xg, yg)
                getgridngp1d(nn, xgsym, pp, nx, self.reducedisinside[:, iya, iza],
                             xmin - gdx[i], xmax - gdx[i])
            elif w3d.solvergeom == w3d.Ygeom:
                xgsym, ygsym = self.applysymmetry(0, yg)
                getgridngp1d(nn, ygsym, pp, ny, self.reducedisinside[ixa, :, iza],
                             ymin - gdy[i], ymax - gdy[i])
            elif w3d.solvergeom == w3d.zgeom:
                getgridngp1d(nn, zgsym, pp, nz, self.reducedisinside[ixa, iya, :],
                             zmin - gdz[i], zmax - gdz[i])

            # --- Loop over the conductors, removing particles that are found inside
            # --- of each.
            for c in self.conductors:

                # --- Get indices relative to the temporary arrays.
                # --- Note that iclose is relative to the full particle arrays.
                itempclose = arange(nn)

                # --- Get indices of particles that are close to the conductor
                ii = compress(pp == c.condid, itempclose)

                # --- If there are no particles close, then skip to the next conductor
                if len(ii) == 0: continue

                # --- Get positions of the particles that are close
                xc = take(xx, ii)
                yc = take(yy, ii)
                zc = take(zz, ii)

                # --- Find the particles that are currently inside and down-select
                # --- the indices. The nint is needed since the quantities is used in
                # --- logical expressions below which require ints.
                xcsym, ycsym = self.applysymmetry(xc, yc)
                currentisinside = nint(c.isinside(xcsym, ycsym, zc).isinside)
                iic = compress(currentisinside, ii)
                ic = take(iclose, iic)

                if self.lrefineallintercept:
                    # --- Refine whether or not particles are lost by taking small time
                    # --- steps, starting from the old position. Note that it is possible
                    # --- that particles that were lost may not be lost upon refinement,
                    # --- and similarly, particles that were not lost, may be lost upon
                    # --- refinement.
                    # --- Get the old coordinates of particles that are close.
                    iclose1 = take(iclose, ii)
                    xo = take(top.pgroup.pid[:, self.xoldpid], iclose1)
                    yo = take(top.pgroup.pid[:, self.yoldpid], iclose1)
                    zo = take(top.pgroup.pid[:, self.zoldpid], iclose1)
                    uxo = take(top.pgroup.pid[:, self.uxoldpid], iclose1)
                    uyo = take(top.pgroup.pid[:, self.uyoldpid], iclose1)
                    uzo = take(top.pgroup.pid[:, self.uzoldpid], iclose1)
                    oldisOK = nint(take(top.pgroup.pid[:, self.oldisOK], iclose1))

                    # --- Get the current fields
                    ex = take(top.pgroup.ex, iclose1)
                    ey = take(top.pgroup.ey, iclose1)
                    ez = take(top.pgroup.ez, iclose1)
                    bx = take(top.pgroup.bx, iclose1)
                    by = take(top.pgroup.by, iclose1)
                    bz = take(top.pgroup.bz, iclose1)

                    # --- This is a possible optmization, which skips particles to
                    # --- far from the conductor to possibly hit it.

                    # --- Get the largest distance that the particles could travel
                    # --- in one time step.
                    qom = top.pgroup.sq[js] / top.pgroup.sm[js]
                    xchange = abs(uxo * top.dt) + abs(0.5 * qom * ex * top.dt ** 2)
                    ychange = abs(uyo * top.dt) + abs(0.5 * qom * ey * top.dt ** 2)
                    zchange = abs(uzo * top.dt) + abs(0.5 * qom * ez * top.dt ** 2)
                    maxchange = sqrt(xchange ** 2 + ychange ** 2 + zchange ** 2)

                    # --- Compare the largest travel distance to the distance from the
                    # --- conductor, and skip particles that are far enough away that
                    # --- they would not hit the conductor. Do a logical_or with
                    # --- currentisinside just to be sure that scraped particles are not
                    # --- missed.
                    distance = c.distance(xo, yo, zo)
                    closeenough = logical_or((maxchange > distance.distance),
                                             currentisinside)

                    # --- Downselect the particles which are close enough to the
                    # --- coductor that they could be lost.
                    ii = compress(closeenough, ii)
                    if len(ii) == 0: continue
                    iclose1 = take(iclose, ii)
                    xc = compress(closeenough, xc)
                    yc = compress(closeenough, yc)
                    zc = compress(closeenough, zc)
                    xo = compress(closeenough, xo)
                    yo = compress(closeenough, yo)
                    zo = compress(closeenough, zo)
                    uxo = compress(closeenough, uxo)
                    uyo = compress(closeenough, uyo)
                    uzo = compress(closeenough, uzo)
                    ex = compress(closeenough, ex)
                    ey = compress(closeenough, ey)
                    ez = compress(closeenough, ez)
                    bx = compress(closeenough, bx)
                    by = compress(closeenough, by)
                    bz = compress(closeenough, bz)
                    currentisinside = compress(closeenough, currentisinside)
                    oldisOK = compress(closeenough, oldisOK)

                    # --- Create some temporaries
                    itime = None
                    dt = top.dt * top.pgroup.ndts[js] * top.pgroup.dtscale[js] * ones(len(ii))
                    q = top.pgroup.sq[js]
                    m = top.pgroup.sm[js]

                    # --- Do the refinement calculation. The currentisinside argument sets
                    # --- when the current position is replaced by the refined position.
                    # --- If the particle is currently lost but in the refined
                    # --- calculation is not lost, then the replace the current position
                    # --- with that refined position that is not lost. Similarly, if the
                    # --- particle is currently not lost, but in the refined calculation
                    # --- is lost, then replace the current position with the refined
                    # --- position.
                    refinedisinside = zeros(len(xc), 'l')
                    self.refineintercept(c, xc, yc, zc, xo, yo, zo, uxo, uyo, uzo,
                                         ex, ey, ez, bx, by, bz, itime, dt, q, m, currentisinside,
                                         refinedisinside)

                    # --- For some newly created particles, there was no old saved data.
                    # --- In those cases, ignore the result of the refined calculation,
                    # --- and use the result obtained originally in currentisinside.
                    refinedisinside = where(oldisOK, refinedisinside, currentisinside)

                    # --- iic lists the particles that are lost in the refined
                    # --- calculation. These will be scraped. Particles which were
                    # --- considered lost but where not lost based on the refined
                    # --- calculation still need to have their refined positions checked
                    # --- against other conductors. There is a possible problem here.
                    # --- The refined trajectory could put the particle in a different
                    # --- grid cell than the original, and it could be inside a conductor
                    # --- that the original wasn't considered close too. This would leave
                    # --- that particle unscraped at that refined position but inside
                    # --- a conductor. This case would be messy to deal with, requiring
                    # --- a second loop over conductors.
                    iic = compress(refinedisinside, ii)
                    ic = take(iclose, iic)

                    # --- Do the replacements as described above. Note that for lost
                    # --- particles, xc,yc,zc hold the positions of the particles one
                    # --- small time step into the conductor. Don't do the replacement
                    # --- for new particles since the old data is no good.
                    iio = (currentisinside | refinedisinside) & oldisOK
                    iiu = compress(iio, arange(shape(xc)[0]))
                    iuserefined = compress(iio, iclose1)
                    put(top.pgroup.xp, iuserefined, take(xc, iiu))
                    put(top.pgroup.yp, iuserefined, take(yc, iiu))
                    put(top.pgroup.zp, iuserefined, take(zc, iiu))
                    put(top.pgroup.uxp, iuserefined, take(uxo, iiu))
                    put(top.pgroup.uyp, iuserefined, take(uyo, iiu))
                    put(top.pgroup.uzp, iuserefined, take(uzo, iiu))

                    # --- Note that the old values of the positions are changed
                    # --- only for particles for which the refined calculation
                    # --- shows they are lost. This is needed for the interception
                    # --- calculation done in savecondid.
                    iclose2 = compress(refinedisinside, iclose1)
                    if len(iclose2) > 0:
                        put(top.pgroup.pid[:, self.xoldpid], iclose2, compress(refinedisinside, xo))
                        put(top.pgroup.pid[:, self.yoldpid], iclose2, compress(refinedisinside, yo))
                        put(top.pgroup.pid[:, self.zoldpid], iclose2, compress(refinedisinside, zo))

                # --- If no particle are inside the conductor, then skip to the next one
                if len(iic) == 0: continue

                if c.material == 'reflector':
                    # --- For lost new particles, which have no old data, not much can be
                    # --- done, so they are set to be removed.
                    oldisOK = nint(take(top.pgroup.pid[:, self.oldisOK], ic))
                    icnew = compress(logical_not(oldisOK), ic)
                    if len(icnew) > 0:
                        put(top.pgroup.gaminv, icnew, 0.)
                        # --- Only old particles will be reflected.
                        ic = compress(oldisOK, ic)

                    # --- For particles which are inside, replace the position with
                    # --- the old position and reverse the velocity.
                    put(top.pgroup.xp, ic, take(top.pgroup.pid[:, self.xoldpid], ic))
                    put(top.pgroup.yp, ic, take(top.pgroup.pid[:, self.yoldpid], ic))
                    put(top.pgroup.zp, ic, take(top.pgroup.pid[:, self.zoldpid], ic))
                    if self.lsaveoldvelocities:
                        # --- If its available, use the old velocity.
                        # --- Should this be the default?
                        put(top.pgroup.uxp, ic, -take(top.pgroup.pid[:, self.uxoldpid], ic))
                        put(top.pgroup.uyp, ic, -take(top.pgroup.pid[:, self.uyoldpid], ic))
                        put(top.pgroup.uzp, ic, -take(top.pgroup.pid[:, self.uzoldpid], ic))
                    else:
                        # --- Otherwise use the new velocity. Can this lead to errors?
                        put(top.pgroup.uxp, ic, -take(top.pgroup.uxp, ic))
                        put(top.pgroup.uyp, ic, -take(top.pgroup.uyp, ic))
                        put(top.pgroup.uzp, ic, -take(top.pgroup.uzp, ic))

                else:
                    # --- For particles which are inside, set gaminv to 0, the lost
                    # --- particle flag
                    if self.directional_scraper:
                        direction_filter = self.directional_scraper * top.pgroup.uzp[
                            ic] < 0  # only pass particles with sign(vz) != sign(directional_scraper)
                        ic = extract(direction_filter, ic)
                    if self.transparency:
                        prob_filter = random.random(ic.shape[0]) > self.transparency
                        ic = extract(prob_filter, ic)
                        put(top.pgroup.gaminv, ic, 0.)
                    else:
                        put(top.pgroup.gaminv, ic, 0.)

                    if c.material == 'dielectric' and self.rhob is not None:
                        np = len(ic)
                        w = self.chargingfactor * top.pgroup.sq[js] * w3d.nx * w3d.nz / (
                                    (w3d.xmmax - w3d.xmmin) * (w3d.zmmax - w3d.zmmin)) * ones(np, 'd')
                        deposgrid2d(1, np, take(top.pgroup.xp, ic), take(top.pgroup.zp, ic), w, w3d.nx, w3d.nz,
                                    self.rhob, self.surfacecount, w3d.xmmin, w3d.xmmax, w3d.zmmin, w3d.zmmax)

                #                        if w3d.boundxy is periodic:
                #                            self.rhob[0,:] = (self.rhob[-1,:] + self.rhob[0,:]) / 1.0
                #                            self.rhob[-1,:] = self.rhob[0,:]
                #
                #                        if w3d.bound0 is periodic:
                #                            self.rhob[:,0] = (self.rhob[:,-1] + self.rhob[:,0]) / 1.0
                #                            self.rhob[:,-1] = self.rhob[:,0]

                # self.surfacespecies.addparticles(x = take(top.pgroup.xp, ic), y = take(top.pgroup.yp, ic), z = take(top.pgroup.zp, ic), vx = 0., vy = 0., vz = 0.)

                # --- Remove the already handled particles, returning if there
                # --- are no more.
                put(iclose, iic, -1)
                iclose = compress(iclose >= 0, iclose)
                nn = len(iclose)
                if nn == 0: return
                put(itempclose, iic, -1)
                itempclose = compress(itempclose >= 0, itempclose)
                xx = take(xx, itempclose)
                yy = take(yy, itempclose)
                zz = take(zz, itempclose)
                pp = take(pp, itempclose)
                if w3d.solvergeom in [w3d.XYZgeom, w3d.XZgeom, w3d.XYgeom, w3d.RZgeom, w3d.Rgeom]:
                    xg = take(xg, itempclose)
                if w3d.solvergeom in [w3d.XYZgeom, w3d.XYgeom, w3d.Ygeom]:
                    yg = take(yg, itempclose)
                if w3d.solvergeom in [w3d.XYZgeom, w3d.XZgeom, w3d.RZgeom, w3d.Zgeom]:
                    zg = take(zg, itempclose)