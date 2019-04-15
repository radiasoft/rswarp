import numpy as np
import time
from itertools import permutations
from warp import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():

    w3d.solvergeom = w3d.XZgeom

    # Set boundary conditions
    # Longitudinal conditions overriden by conducting plates
    w3d.bound0  = dirichlet
    w3d.boundnz = dirichlet
    w3d.boundxy = periodic

    # Particles boundary conditions
    top.pbound0  = absorb
    top.pboundnz = absorb
    top.pboundxy = periodic

    w3d.xmmin = -0.7
    w3d.xmmax =  0.7
    w3d.zmmin =  0.0
    w3d.zmmax =  1.4

    w3d.nx = 140
    w3d.nz = 140

    # Particles
    top.inject = 1
    beam = Species(type=Electron, name='beam')
    w3d.l_inj_exact = False

    PTCL_PER_STEP = 100
    top.npinject = PTCL_PER_STEP
    w3d.l_inj_addtempz_abs = True
    w3d.l_inj_exact = True
    top.linj_rectangle = (w3d.solvergeom == w3d.XZgeom)
    w3d.l_inj_rz = (w3d.solvergeom == w3d.RZgeom)

    beam.a0       = w3d.xmmax
    beam.b0       = w3d.xmmax
    beam.ap0      = .0e0
    beam.bp0      = .0e0
    beam.ibeam    = 1e-3
    beam.vzinject = 1.4 / 140 / 1e-10
    beam.vthperp  = 0
    derivqty()


    # Conductors
    p = ZPlane(voltage=0, zcent=w3d.zmmax)
    b1 = Box(xsize=0.5, ysize=1.0, zsize=0.7, xcent=-0.35, ycent=0.35, zcent=0.75, voltage=+39.0)
    b2 = Box(xsize=0.3, ysize=1.0, zsize=0.3, xcent=0.35, ycent=0.35, zcent=0.55, voltage=+33.0)
    installed_conductors = [p, b1, b2]

    E = MultiGrid2D()
    registersolver(E)
    for cond in installed_conductors:
        E.installconductor(cond)
    scraper = ParticleScraper(installed_conductors, lcollectlpdata=True)

    # Run
    top.dt = 8e-11
    top.lprntpara = False
    top.lpsplots = False
    package("w3d")
    generate()
    for _ in range(1):
        step(150)


    # Plotting
    from rswarp.diagnostics.ImpactDensity import PlotDensity
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize = (12, 8))

    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 0.1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    myplot = PlotDensity(ax1, ax2, scraper=scraper, top=top, w3d=w3d, interpolation='kde')
    myplot.generate_plots_2d()
    plt.show()


if __name__ == "__main__":
    main()
