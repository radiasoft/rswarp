# From grid_field_testing.py

from __future__ import division

# Set matplotlib backend for saving plots (if requested)
import matplotlib
matplotlib.use('PS')

# set warpoptions.ignoreUnknownArgs = True before main import to allow command line arguments
import warpoptions
warpoptions.ignoreUnknownArgs = True

from warp import *

import numpy as np
import h5py as h5
import time
import sys

sys.path.append('/global/homes/h/hallcc/github/rswarp')

from copy import deepcopy
from random import randint
from rswarp.cathode import sources
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.file_utils import cleanupPrevious
from rswarp.diagnostics.parallel import save_lost_particles
from rswarp.diagnostics.ConductorDiagnostics import analyze_scraped_particles

# Constants imports
from scipy.constants import e, m_e, c, k

kb_eV = 8.6173324e-5  # Bolztmann constant in eV/K
kb_J = k  # Boltzmann constant in J/K
m = m_e  # electron mass


def main(x_struts, y_struts, volts_on_grid, grid_height, strut_width, strut_height, id,
         injection_type=1, cathode_temperature=1273.15,
         random_seed=True, install_grid=True,
         particle_diagnostic_switch=False, field_diagnostic_switch=False, lost_diagnostic_flag=False):
    # record inputs
    run_attributes = deepcopy(locals())

    # set new random seed
    if random_seed:
        top.seedranf(randint(1, 1e9))

    # Control for printing in parallel
    if comm_world.size != 1:
        synchronizeQueuedOutput_mpi4py(out=True, error=True)

    if particle_diagnostic_switch or field_diagnostic_switch:
        # Directory paths
        diagDir = 'diags_id{}/xyzsolver/hdf5/'.format(id)
        field_base_path = 'diags_id{}/fields/'.format(id)
        diagFDir = {'magnetic': 'diags_id{}/fields/magnetic'.format(id),
                    'electric': 'diags_id{}/fields/electric'.format(id)}

        # Cleanup command if directories already exist
        if comm_world.rank == 0:
            cleanupPrevious(diagDir,diagFDir)

    ######################
    # DOMAIN/GEOMETRY/MESH
    ######################

    PLATE_SPACING = 1e-6  # plate spacing
    CHANNEL_WIDTH = 100e-9  # width of simulation box

    # Dimensions
    X_MAX = +CHANNEL_WIDTH / 2.
    X_MIN = -X_MAX
    Y_MAX = +CHANNEL_WIDTH / 2.
    Y_MIN = -Y_MAX
    Z_MAX = PLATE_SPACING
    Z_MIN = 0.

    # Grid parameters
    NUM_X = 100
    NUM_Y = 100
    NUM_Z = 1000

    # z step size
    dz = (Z_MAX - Z_MIN) / NUM_Z

    # Solver Geometry and Boundaries

    # Specify solver geometry
    w3d.solvergeom = w3d.XYZgeom

    # Set field boundary conditions
    w3d.bound0 = neumann
    w3d.boundnz = dirichlet
    w3d.boundxy = periodic
    # Particles boundary conditions
    top.pbound0 = absorb
    top.pboundnz = absorb
    top.pboundxy = periodic

    # Set mesh boundaries
    w3d.xmmin = X_MIN
    w3d.xmmax = X_MAX
    w3d.ymmin = Y_MIN
    w3d.ymmax = Y_MAX
    w3d.zmmin = 0.
    w3d.zmmax = Z_MAX

    # Set mesh cell counts
    w3d.nx = NUM_X
    w3d.ny = NUM_Y
    w3d.nz = NUM_Z

    #############################
    # PARTICLE INJECTION SETTINGS
    #############################

    # INJECTION SPECIFICATION
    USER_INJECT = injection_type

    # Cathode and anode settings
    CATHODE_TEMP = cathode_temperature
    CATHODE_PHI = 2.0  # work function in eV
    ANODE_WF = 0.1  # Can be used if vacuum level is being set
    ACCEL_VOLTS = volts_on_grid  # ACCEL_VOLTS used for velocity and CL calculations

    # Emitted species
    beam = Species(type=Electron, name='beam')

    # Emitter area and position
    SOURCE_RADIUS_1 = 0.5 * CHANNEL_WIDTH  # a0 parameter - X plane
    SOURCE_RADIUS_2 = 0.5 * CHANNEL_WIDTH  # b0 parameter - Y plane
    Z_PART_MIN = dz / 1000.  # starting particle z value

    # Compute cathode area for geomtry-specific current calculations
    if (w3d.solvergeom == w3d.XYZgeom):
        # For 3D cartesion geometry only
        cathode_area = 4. * SOURCE_RADIUS_1 * SOURCE_RADIUS_2
    else:
        # Assume 2D XZ geometry
        cathode_area = 2. * SOURCE_RADIUS_1 * 1.

    # If using the XZ geometry, set so injection uses the same geometry
    top.linj_rectangle = (w3d.solvergeom == w3d.XZgeom or w3d.solvergeom == w3d.XYZgeom)

    # Returns velocity beam_beta (in units of beta) for which frac of emitted particles have v < beam_beta * c
    beam_beta = sources.compute_cutoff_beta(CATHODE_TEMP, frac=0.99)

    PTCL_PER_STEP = 300
    CURRENT_MODIFIER = 0.5  # Factor to reduce current by


    if USER_INJECT == 1:
        # Constant current density - beam transverse velocity fixed to zero, very small longitduinal velocity

        # Set injection flag
        top.inject = 1  # 1 means constant; 2 means space-charge limited injection;# 6 means user-specified
        top.npinject = PTCL_PER_STEP
        beam_current = 4. / 9. * eps0 * sqrt(2. * echarge / beam.mass) \
                       * ACCEL_VOLTS ** 1.5 / PLATE_SPACING ** 2 * cathode_area

        beam.ibeam = beam_current * CURRENT_MODIFIER

        beam.a0 = CHANNEL_WIDTH
        beam.b0 = CHANNEL_WIDTH
        beam.ap0 = .0e0
        beam.bp0 = .0e0

        w3d.l_inj_exact = True

        # Initial velocity settings (5% of c)
        vrms = np.sqrt(1 - 1 / (0.05 / 511e3 + 1) ** 2) * 3e8
        top.vzinject = vrms

    if USER_INJECT == 2:
        # True Thermionic injection
        top.inject = 1
        top.npinject = PTCL_PER_STEP

        beam_current = 4. / 9. * eps0 * sqrt(2. * echarge / beam.mass) \
                       * ACCEL_VOLTS ** 1.5 / PLATE_SPACING ** 2 * cathode_area

        beam.ibeam = beam_current * CURRENT_MODIFIER

        beam.a0 = CHANNEL_WIDTH
        beam.b0 = CHANNEL_WIDTH
        beam.ap0 = .0e0
        beam.bp0 = .0e0

        w3d.l_inj_exact = True

        # Specify thermal properties
        beam.vthz = np.sqrt(CATHODE_TEMP * kb_J / beam.mass)
        beam.vthperp = np.sqrt(CATHODE_TEMP * kb_J / beam.mass)
        top.lhalfmaxwellinject = 1  # inject z velocities as half Maxwellian

        beam_current = sources.j_rd(CATHODE_TEMP, CATHODE_PHI) * cathode_area  # steady state current in Amps
        beam.ibeam = beam_current * CURRENT_MODIFIER
        beam.a0 = SOURCE_RADIUS_1
        beam.b0 = SOURCE_RADIUS_2

    derivqty()

    ##############
    # FIELD SOLVER
    ##############

    # Set up fieldsolver
    f3d.mgtol = 1e-6
    solverE = MultiGrid3D()
    registersolver(solverE)

    ########################
    # CONDUCTOR INSTALLATION
    ########################

    if install_grid:
        accel_grid, gl = create_grid(x_struts, y_struts, volts_on_grid,
                                     grid_height, strut_width, strut_height,
                                     CHANNEL_WIDTH)
        accel_grid.voltage = volts_on_grid

    # --- Anode Location
    zplate = Z_MAX

    # Create source conductors
    source = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0., condid=2)
    # Create ground plate
    if install_grid:
        plate = ZPlane(voltage=0., zcent=zplate, condid=3)
    else:
        plate = ZPlane(voltage=volts_on_grid, zcent=zplate, condid=3)
    #####
    # Grid dimensions used in testing
    # strut_width = 3e-9
    # strut_height = 30e-9
    # grid_height = zplate / 2.
    # volts_on_grid = 5.

    # print "Initial Grid id:", accel_grid.condid, source.condid, plate.condid

    if install_grid:
        installconductor(accel_grid)
        installconductor(source, dfill=largepos)
        installconductor(plate, dfill=largepos)
        scraper = ParticleScraper([source, plate, accel_grid],
                                  lcollectlpdata=True,
                                  lsaveintercept=True)
        scraper_dictionary = {1: 'grid', 2: 'source', 3: 'collector'}
    else:
        installconductor(source, dfill=largepos)
        installconductor(plate, dfill=largepos)
        scraper = ParticleScraper([source, plate])
        scraper_dictionary = {2: 'source', 3: 'collector'}

    #############
    # DIAGNOSTICS
    #############

    # Particle/Field diagnostic options
    if particle_diagnostic_switch:
        particleperiod = 1000
        particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=top, w3d=w3d,
                                                   species={species.name: species for species in listofallspecies},
                                                   comm_world=comm_world, lparallel_output=False,
                                                   write_dir=diagDir[:-5])
        installafterstep(particle_diagnostic_0.write)

    if field_diagnostic_switch:
        fieldperiod = 1000
        efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d,
                                                                  comm_world=comm_world,
                                                                  period=fieldperiod)
        installafterstep(efield_diagnostic_0.write)

    def get_lost_counts():
        scraper_record = analyze_scraped_particles(top, beam, solverE)

        for key in scraper_dictionary:
            print 'Step: {}'.format(top.it)
            print 'analyze says {} scraped:'.format(scraper_dictionary[key]), \
                np.sum(scraper_record[key][:, 1])
        if install_grid:
            return np.array([np.sum(scraper_record[1][:, 1]), np.sum(scraper_record[2][:, 1]),
                             np.sum(scraper_record[3][:, 1])])
        else:
            return np.array([np.sum(scraper_record[2][:, 1]),
                             np.sum(scraper_record[3][:, 1])])
    ##########################
    # SOLVER SETTINGS/GENERATE
    ##########################

    # prevent gist from starting upon setup
    top.lprntpara = false
    top.lpsplots = false

    top.verbosity = 0  # Reduce solver verbosity
    solverE.mgverbose = 0  # further reduce output upon stepping - prevents websocket timeouts in Jupyter notebook

    init_iters = 20000
    regular_iters = 200

    init_tol = 1e-6
    regular_tol = 1e-6

    # Time Step

    # Determine an appropriate time step based upon estimated final velocity
    vzfinal = sqrt(2. * abs(volts_on_grid) * np.abs(beam.charge) / beam.mass) + beam_beta * c
    dt = dz / vzfinal
    top.dt = dt

    solverE.mgmaxiters = init_iters
    solverE.mgtol = init_tol
    package("w3d")
    generate()
    solverE.mgtol = regular_tol
    solverE.mgmaxiters = regular_iters

    ##################
    # CONTROL SEQUENCE
    ##################

    times = []  # Write out timing of cycle steps to file
    clock = 0  # clock tracks if the simulation has run too long and needs to be terminated

    # Run for 1000 steps initially
    time1 = time.time()
    if install_grid:
        counts_0 = np.array([0., 0., 0.])
    else:
        counts_0 = np.array([0., 0.])
    step(1000)

    counts_1 = get_lost_counts()

    # START TEMP

    for cond in solverE.conductordatalist:
        print "SOLVER COND DATA"
        print cond[0].condid, beam.sw, e
        print cond[0].lostparticles_data[()][:, 0].shape
        print np.sum(cond[0].lostparticles_data[()][:, 1])
        print np.sum(np.round(-1. * cond[0].lostparticles_data[()][:, 1] / beam.sw / e))
    for cond in scraper.conductors:
        print "SCRAPER COND DATA"
        print cond.condid, beam.sw, e
        print cond.lostparticles_data[()][:, 0].shape
        print np.sum(cond.lostparticles_data[()][:, 1])
        print np.sum(np.round(-1. * cond.lostparticles_data[()][:, 1] / beam.sw / e))

    # END TEMP

    time2 = time.time()
    times.append(time2 - time1)
    clock += times[-1]

    # Start run cycle

    tol = 1e9  # Large initial error for check
    target = 0.1  # Final tolerance we are trying to reach
    time_limit = 1.5 * 60 * 60  # 1.5 hour time limit

    while (tol > target) and (clock < time_limit):
        if lost_diagnostic_flag:
            lost_particle_file = 'lost_particles_id{}_step{}.npy'.format(id, top.it)
            save_lost_particles(top, comm_world, fsave=lost_particle_file)

        time1 = time.time()

        step(1000)

        counts_2 = get_lost_counts()

        # START TEMP

        for cond in solverE.conductordatalist:
            print "SOLVER COND DATA"
            print cond[0].condid, beam.sw, e
            print cond[0].lostparticles_data[()][:, 0].shape
            print np.sum(cond[0].lostparticles_data[()][:, 1])
            print np.sum(np.round(-1. * cond[0].lostparticles_data[()][:, 1] / beam.sw / e))
        for cond in scraper.conductors:
            print "SCRAPER COND DATA"
            print cond.condid, beam.sw, e
            print cond.lostparticles_data[()][:, 0].shape
            print np.sum(cond.lostparticles_data[()][:, 1])
            print np.sum(np.round(-1. * cond.lostparticles_data[()][:, 1] / beam.sw / e))

        # END TEMP

        # Record i-1 and i+1 intervals
        accumulated_0 = counts_1 - counts_0
        accumulated_1 = counts_2 - counts_1

        if counts_2[-1] < 1000:
            # Insufficient statistics to start counting, keep tol at default
            collector_fraction_0 = -1.
            collector_fraction_1 = -1.
        else:
            collector_fraction_0 = accumulated_0[-1] / np.sum(accumulated_0,
                                                             dtype=float)  # Fraction of particles incident on collector
            collector_fraction_1 = accumulated_1[-1] / np.sum(accumulated_1,
                                                             dtype=float)  # Fraction of particles incident on collector
            tol = abs(collector_fraction_1 - collector_fraction_0) / collector_fraction_1

        # Re-index values, counts_2 redefined at start of next loop
        counts_0, counts_1 = counts_1, counts_2

        time2 = time.time()
        times.append(time2 - time1)
        clock += times[-1]

    ######################
    # FINAL RUN STATISTICS
    ######################

    if comm_world.rank == 0:
        with open('output_stats_id{}.txt'.format(id), 'w') as f1:
            for ts in times:
                f1.write('{}\n'.format(ts))
            f1.write('\n')
            f1.write('{} {} {} {}\n'.format(collector_fraction_0, accumulated_0[0], accumulated_0[1], accumulated_0[2]))
            f1.write('{} {} {} {}\n'.format(collector_fraction_1, accumulated_1[0], accumulated_1[1], accumulated_1[2]))

        filename = 'efficiency_id{}.h5'.format(str(id))
        with h5.File(filename, 'w') as h5file:
            for key in run_attributes:
                h5file.attrs[key] = run_attributes[key]
            h5file.attrs['grid'] = counts_2[0]
            h5file.attrs['source'] = counts_2[1]
            h5file.attrs['collector'] = counts_2[2]


def create_grid(nx, ny, volts,
                grid_height, strut_width, strut_height,
                strut_length):
    grid_list = []
    positions_x = np.linspace(w3d.xmmin, w3d.xmmax, 2 * nx + 1)
    positions_y = np.linspace(w3d.ymmin, w3d.ymmax, 2 * ny + 1)
    for i in positions_y[1:-1:2]:
        box = Box(strut_length, strut_width, strut_height,
                  xcent=0., ycent=i, zcent=grid_height,
                  voltage=volts,
                  condid='next')
        grid_list.append(box)
    for j in positions_x[1:-1:2]:
        box = Box(strut_width, strut_length, strut_height,
                  xcent=j, ycent=0., zcent=grid_height,
                  voltage=volts,
                  condid='next')
        grid_list.append(box)

    grid = grid_list[0]
    for strut in grid_list[1:]:
        grid += strut

    return grid, grid_list