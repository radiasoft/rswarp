from __future__ import division

# TODO: cycle times for comparison against wall clock are still not calculated correctly
# TODO: decide on handling for tails. are we hurting the effiency estimation?
# set `warpoptions.ignoreUnknownArgs = True` before main import to allow command line arguments Warp does not recognize
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
from scipy.signal import lfilter
from rswarp.cathode import sources
from rswarp.run_files import efficiency
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.particles.extpart import ZCrossingParticles
from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.file_utils import cleanupPrevious
from rswarp.diagnostics.parallel import save_lost_particles
from rswarp.diagnostics.ConductorDiagnostics import analyze_scraped_particles

# Constants imports
from scipy.constants import e, m_e, c, k

kb_eV = 8.6173324e-5  # Bolztmann constant in eV/K
kb_J = k  # Boltzmann constant in J/K
m = m_e  # electron mass in kg


def main(x_struts, y_struts, V_grid, grid_height, strut_width, strut_height,
         rho_ew, T_em, phi_em, T_coll, phi_coll, rho_cw, gap_distance,
         run_id,
         injection_type=2, random_seed=True, install_grid=True, max_wall_time=0.,
         particle_diagnostic_switch=False, field_diagnostic_switch=False, lost_diagnostic_switch=False):
    """
    Run a simulation of a gridded TEC.
    Args:
        x_struts: Number of struts that intercept the x-axis.
        y_struts: Number of struts that intercept the y-axis
        V_grid: Voltage to place on the grid in Volts.
        grid_height: Distance from the emitter to the grid normalized by gap_distance, unitless.
        strut_width: Transverse extent of the struts in meters.
        strut_height: Longitudinal extent of the struts in meters.
        run_id: Run ID. Mainly used for parallel optimization.
        injection_type: 1: For constant current emission with only thermal velocity spread in z and CL limited emission.
                        2: For true thermionic emission. Velocity spread along all axes.
        cathode_temperature: Temperature of emitter. Current density is governed by Richardson-Dushman.
        random_seed: True/False. If True, will force a random seed to be used for emission positions.
        install_grid: True/False. If False the grid will not be installed. Results in simple parallel plate setup.
                                  If False then voltage_on_grid specifies the voltage on the collector.
        particle_diagnostic_switch: True/False. Use openPMD compliant .h5 particle diagnostics.
        field_diagnostic_switch: True/False. Use rswarp electrostatic .h5 field diagnostics (Maybe openPMD compliant?).
        lost_diagnostic_switch: True/False. Enable collection of lost particle coordinates
                        with rswarp.diagnostics.parallel.save_lost_particles.

    """
    # record inputs and set parameters
    run_attributes = deepcopy(locals())

    for key in run_attributes:
        if key in efficiency.tec_parameters:
            efficiency.tec_parameters[key][0] = run_attributes[key]

    # set new random seed
    if random_seed:
        top.seedranf(randint(1, 1e9))

    # Control for printing in parallel
    if comm_world.size != 1:
        synchronizeQueuedOutput_mpi4py(out=True, error=True)

    if particle_diagnostic_switch or field_diagnostic_switch:
        # Directory paths
        diagDir = 'diags_id{}/hdf5/'.format(run_id)
        field_base_path = 'diags_id{}/fields/'.format(run_id)
        diagFDir = {'magnetic': 'diags_id{}/fields/magnetic'.format(run_id),
                    'electric': 'diags_id{}/fields/electric'.format(run_id)}

        # Cleanup command if directories already exist
        if comm_world.rank == 0:
            cleanupPrevious(diagDir, diagFDir)

    ######################
    # DOMAIN/GEOMETRY/MESH
    ######################

    CHANNEL_WIDTH = 100e-9  # width of simulation box

    # Dimensions
    X_MAX = +CHANNEL_WIDTH / 2.
    X_MIN = -X_MAX
    Y_MAX = +CHANNEL_WIDTH / 2.
    Y_MIN = -Y_MAX
    Z_MAX = gap_distance
    Z_MIN = 0.

    # Grid parameters
    NUM_X = 100
    NUM_Y = 100
    NUM_Z = int(gap_distance / 1e-9)

    # z mesh spacing
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

    # Cathode and anode settings
    EMITTER_TEMP = T_em
    EMITTER_PHI = phi_em # work function in eV
    COLLECTOR_PHI = phi_coll  # Can be used if vacuum level is being set
    ACCEL_VOLTS = V_grid  # ACCEL_VOLTS used for velocity and CL calculations

    # Emitted species
    background_beam = Species(type=Electron, name='background')
    measurement_beam = Species(type=Electron, name='measurement')

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
    beam_beta = sources.compute_cutoff_beta(EMITTER_TEMP, frac=0.99)

    PTCL_PER_STEP = 300

    if injection_type == 1:
        CURRENT_MODIFIER = 0.5  # Factor to multiply CL current by when setting beam current
        # Constant current density - beam transverse velocity fixed to zero, very small longitduinal velocity

        # Set injection flag
        top.inject = 1  # 1 means constant; 2 means space-charge limited injection;# 6 means user-specified
        top.npinject = PTCL_PER_STEP
        beam_current = 4. / 9. * eps0 * sqrt(2. * echarge / background_beam.mass) \
                       * ACCEL_VOLTS ** 1.5 / gap_distance ** 2 * cathode_area

        background_beam.ibeam = beam_current * CURRENT_MODIFIER

        background_beam.a0 = SOURCE_RADIUS_1
        background_beam.b0 = SOURCE_RADIUS_1
        background_beam.ap0 = .0e0
        background_beam.bp0 = .0e0

        w3d.l_inj_exact = True

        # Initial velocity settings (5% of c)
        vrms = np.sqrt(1 - 1 / (0.05 / 511e3 + 1) ** 2) * 3e8
        top.vzinject = vrms

    if injection_type == 2:
        # True Thermionic injection
        top.inject = 1

        # Set both beams to same npinject to keep weights the same
        background_beam.npinject = PTCL_PER_STEP
        measurement_beam.npinject = PTCL_PER_STEP

        # Use rnpinject to set number of macroparticles emitted
        background_beam.rnpinject = PTCL_PER_STEP
        measurement_beam.rnpinject = 0  # measurement beam is off at start

        w3d.l_inj_exact = True

        # Specify thermal properties
        background_beam.vthz = measurement_beam.vthz = np.sqrt(EMITTER_TEMP * kb_J / background_beam.mass)
        background_beam.vthperp = measurement_beam.vthperp = np.sqrt(EMITTER_TEMP * kb_J / background_beam.mass)
        top.lhalfmaxwellinject = 1  # inject z velocities as half Maxwellian

        beam_current = sources.j_rd(EMITTER_TEMP, EMITTER_PHI) * cathode_area  # steady state current in Amps
        print('beam current expected: {}, current density {}'.format(beam_current, beam_current / cathode_area))
        background_beam.ibeam = measurement_beam.ibeam = beam_current
        background_beam.a0 = measurement_beam.a0 = SOURCE_RADIUS_1
        background_beam.b0 = measurement_beam.b0 = SOURCE_RADIUS_1
        background_beam.ap0 = measurement_beam.ap0 = .0e0
        background_beam.bp0 = measurement_beam.bp0 = .0e0

    derivqty()
    print('weight {}'.format(background_beam.sw))
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
        accel_grid, gl = create_grid(x_struts, y_struts, V_grid,
                                     grid_height * gap_distance, strut_width, strut_height,
                                     CHANNEL_WIDTH)
        accel_grid.voltage = V_grid

    # --- Anode Location
    zplate = Z_MAX

    # Create source conductors
    if install_grid:
        source = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0., condid=2)
    else:
        source = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0.)
    # Create ground plate
    if install_grid:
        plate = ZPlane(voltage=0., zcent=zplate, condid=3)
    else:
        plate = ZPlane(voltage=V_grid, zcent=zplate)
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
        scraper = ParticleScraper([accel_grid, source, plate],
                                  lcollectlpdata=True,
                                  lsaveintercept=True)
        scraper_dictionary = {'grid': 1, 'source': 2, 'collector': 3}
    else:
        installconductor(source, dfill=largepos)
        installconductor(plate, dfill=largepos)
        scraper = ParticleScraper([source, plate],
                                  lcollectlpdata=True,
                                  lsaveintercept=True)
        scraper_dictionary = {'source': 1, 'collector': 2}

    #############
    # DIAGNOSTICS
    #############

    # Particle/Field diagnostic options
    if particle_diagnostic_switch:
        particleperiod = 100000
        particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=top, w3d=w3d,
                                                   species={species.name: species
                                                            for species in listofallspecies
                                                            if species.name == 'measurement'},
                                                   comm_world=comm_world, lparallel_output=False,
                                                   write_dir=diagDir[:-5])
        installafterstep(particle_diagnostic_0.write)

    if field_diagnostic_switch:
        fieldperiod = 1000
        efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d,
                                                                  comm_world=comm_world,
                                                                  period=fieldperiod)
        installafterstep(efield_diagnostic_0.write)

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
    vz_accel = sqrt(2. * abs(V_grid) * np.abs(background_beam.charge) / background_beam.mass)
    vzfinal = vz_accel + beam_beta * c
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
    # Run until steady state is achieved (flat current profile at collector) (measurement species turned on)
    # Record data for effiency calculation
    # Switch off measurement species and wait for simulation to clear (background species is switched on)

    early_abort = False  # If true will flag output data to notify
    startup_time = 2 * gap_distance / vz_accel  # Roughly 2 crossing times for system to reach steady state
    crossing_measurements = 8  # Number of crossing times to record for
    steps_per_crossing = gap_distance / vzfinal / dt
    ss_check_interval = int(steps_per_crossing / 5.)
    times = []  # Write out timing of cycle steps to file
    clock = 0  # clock tracks if the simulation has run too long and needs to be terminated

    # Run initial block of steps
    record_time(stept, times, startup_time)
    clock += times[-1]

    print("Completed Initialization on Step {}\n Initialization run time: {}".format(top.it, times[-1]))

    # Start checking for Steady State Operation
    tol = 0.05
    steady_state = 0
    while steady_state != 1:
        record_time(step, times, ss_check_interval)
        clock += times[-1]
        steady_state, avg, stdev = stead_state_check(background_beam, solverE,
                                                     scraper_dictionary['collector'], ss_check_interval, tol=tol)
        print(" For {} steps: Avg particles/step={}, Stdev particles/step={}, tol={}".format(ss_check_interval,
                                                                                             avg, stdev, tol))

    # Start Steady State Operation
    print(" Steady State Reached.\n Starting efficiency "
          "recording for {} crossing times.\n This will be {} steps".format(crossing_measurements,
                                                                            steps_per_crossing * crossing_measurements))
    particle_diagnostic_0.period = steps_per_crossing
    # Switch to measurement beam species
    measurement_beam.rnpinject = PTCL_PER_STEP
    background_beam.rnpinject = 0

    # Install Zcrossing Diagnostic
    # TODO: make change to grid_height fractional
    ZCross = ZCrossingParticles(zz=grid_height * gap_distance / 200., laccumulate=1)
    emitter_flux = []

    crossing_wall_time = times[-1] * steps_per_crossing / ss_check_interval  # Estimate wall time for one crossing
    print('crossing_wall_time estimate: {}, for {} steps'.format(crossing_wall_time, steps_per_crossing))
    print('wind-down loop time estimate: {}, for {} steps'.format(crossing_wall_time * steps_per_crossing / ss_check_interval, ss_check_interval))
    for sint in range(crossing_measurements):
        # Kill the loop and proceed to writeout if we don't have time to complete the loop
        if (max_wall_time - clock) < crossing_wall_time:
            early_abort = True
            break

        record_time(step, times, steps_per_crossing)
        clock += times[-1]

        emitter_flux.append(np.array([ZCross.getvx(js=measurement_beam.js),
                             ZCross.getvy(js=measurement_beam.js),
                             ZCross.getvz(js=measurement_beam.js)]).transpose())
        ZCross.clear()  # Clear ZcrossingParticles memory

        print("Measurement: {} of {} intervals completed. Interval run time: {} s".format(sint + 1,
                                                                                          crossing_measurements,
                                                                                          times[-1]))

    # Run wind-down until measurement particles have cleared
    measurement_beam.rnpinject = 0
    background_beam.rnpinject = PTCL_PER_STEP

    initial_population = measurement_beam.npsim[0]
    measurement_tol = 0.075
    particle_diagnostic_0.period = ss_check_interval
    while measurement_beam.npsim[0] / initial_population > measurement_tol:
        # Kill the loop and proceed to writeout if we don't have time to complete the loop
        if (max_wall_time - clock) < crossing_wall_time * ss_check_interval / steps_per_crossing :
            early_abort = True
            break

        record_time(step, times, ss_check_interval)
        clock += times[-1]

        if ZCross.getvx(js=measurement_beam.js).shape[0] > 0:
            emitter_flux.append(np.array([ZCross.getvx(js=measurement_beam.js),
                                ZCross.getvy(js=measurement_beam.js),
                                ZCross.getvz(js=measurement_beam.js)]).transpose())
            ZCross.clear()  # Clear ZcrossingParticles memory
        print "Backwards particles: {}".format(np.where(emitter_flux[-1][:, 2] < 0.)[0].shape)
        print(" Wind-down: Taking {} steps, On Step: {}, {} Particles Left".format(ss_check_interval, top.it,
                                                                                   measurement_beam.npsim[0]))
    ######################
    # CALCULATE EFFICIENCY
    ######################
    emitter_flux = np.vstack(emitter_flux)

    # Find integrated charge on each conductor
    surface_charge = analyze_scraped_particles(top, measurement_beam, solverE)
    measured_charge = {}

    for key in surface_charge:
        # We can abuse the fact that js=0 for background species to filter it from the sum
        measured_charge[key] = np.sum(surface_charge[key][:, 1] * surface_charge[key][:, 3])

    # Set externally derived parameters
    efficiency.tec_parameters['A_em'][0] = cathode_area * 1e4  # cm**2
    efficiency.tec_parameters['occlusion'][0] = efficiency.calculate_occlusion(**efficiency.tec_parameters)

    # Set derived parameters from simulation
    efficiency.tec_parameters['run_time'][0] = crossing_measurements * steps_per_crossing * dt
    efficiency.tec_parameters['J_em'][0] = e * (emitter_flux.shape[0] - measured_charge[scraper_dictionary['source']]) \
                                        * measurement_beam.sw / \
                                        efficiency.tec_parameters['run_time'][0] / efficiency.tec_parameters['A_em'][0]
    efficiency.tec_parameters['J_grid'][0] = e * measured_charge[scraper_dictionary['grid']] * measurement_beam.sw / \
                                        efficiency.tec_parameters['run_time'][0] / \
                                        (efficiency.tec_parameters['occlusion'][0] *
                                         efficiency.tec_parameters['A_em'][0])
    efficiency.tec_parameters['J_ec'][0] = e * measured_charge[scraper_dictionary['collector']] * measurement_beam.sw / \
                                        efficiency.tec_parameters['run_time'][0] / efficiency.tec_parameters['A_em'][0]
    efficiency.tec_parameters['P_em'][0] = efficiency.calculate_power_flux(emitter_flux, measurement_beam.sw,
                                                                        efficiency.tec_parameters['phi_em'][0],
                                                                        **efficiency.tec_parameters)

    # Efficiency calculation
    print "Efficiency", efficiency.calculate_efficiency(**efficiency.tec_parameters)

    ######################
    # FINAL RUN STATISTICS
    ######################

    import pickle

    if comm_world.rank == 0:
        pickle.dump(surface_charge, open("test_current_data.p", "wb"))
        pickle.dump(efficiency.tec_parameters, open("test_parameter_data.p", "wb"))
        with open('output_stats_id{}.txt'.format(run_id), 'w') as f1:
            for ts in times:
                f1.write('{}\n'.format(ts))
            if early_abort:
                f1.write('ABORTED')
            f1.write('\n')

        filename = 'efficiency_id{}.h5'.format(str(run_id))
        with h5.File(filename, 'w') as h5file:
            h5file.attrs['complete'] = early_abort
            for key in run_attributes:
                h5file.attrs[key] = run_attributes[key]
            for key, value in scraper_dictionary.iteritems():
                h5file.attrs[key] = measured_charge[value]


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


def record_time(func, time_list, *args, **kwargs):
    t1 = time.time()
    func(*args, **kwargs)
    t2 = time.time()
    time_list.append(t2 - t1)


def stead_state_check(particles, solver, sid, interval, tol=0.01, n=185, a=1):
    b = [1.0 / n] * n
    collector_current = analyze_scraped_particles(top, particles, solver)[sid]
    y = lfilter(b, a, collector_current[:, 1])[-interval:]  # Assumes that charge is being deposited ~every step
    avg = np.average(y)
    stdev = np.std(y)
    if stdev / avg < tol:
        ss = 1
    else:
        ss = 0

    return ss, avg, stdev
