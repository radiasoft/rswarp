from __future__ import division

# TODO: cycle times for comparison against wall clock are still not calculated correctly
# TODO: decide on handling for tails. are we hurting the effiency estimation?
# set `warpoptions.ignoreUnknownArgs = True` before main import to allow command line arguments Warp does not recognize
import warpoptions
warpoptions.ignoreUnknownArgs = True

from warp import *

import numpy as np
import h5py as h5
import sys, os

sys.path.append('/global/homes/h/hallcc/github/rswarp')

from copy import deepcopy
from random import randint
from rswarp.cathode import sources
from rswarp.run_files.tec import efficiency
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.particles.extpart import ZCrossingParticles
from warp.utils.loadbalance import LoadBalancer
from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.file_utils import cleanupPrevious
from tec_utilities import record_time, SteadyState, ExternalCircuit, write_parameter_file
from rswarp.diagnostics.ConductorDiagnostics import analyze_scraped_particles

# Constants imports
from scipy.constants import e, m_e, c, k

kb_eV = 8.6173324e-5  # Bolztmann constant in eV/K
kb_J = k  # Boltzmann constant in J/K
m = m_e  # electron mass in kg


def main(x_struts, y_struts, V_grid, grid_height, strut_width, strut_height,
         rho_ew, T_em, phi_em, T_coll, phi_coll, rho_cw, gap_distance, rho_load,
         run_id, channel_width=100e-9,
         injection_type=2, magnetic_field=0.0, random_seed=True, install_grid=True, max_wall_time=1e9,
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
        rho_ew: Emitter side wiring resistivity, ohms*cm.
        T_em: Emitter temperature, kelvin.
        phi_em: Emitter work function, eV.
        T_coll: Collector termperature, kelvin.
        phi_coll: Collector work function, eV.
        rho_cw: Collector side wiring resistivity, ohms*cm.
        gap_distance: Distance from emitter to collector, meters.
        rho_load: Load resistivity, ohms*cm.
        run_id: Run ID. Mainly used for parallel optimization.
        injection_type: 1: For constant current emission with only thermal velocity spread in z and CL limited emission.
                        2: For true thermionic emission. Velocity spread along all axes.
        random_seed: True/False. If True, will force a random seed to be used for emission positions.
        install_grid: True/False. If False the grid will not be installed. Results in simple parallel plate setup.
                                  If False then phi_em - phi_coll specifies the voltage on the collector.
        max_wall_time: Wall time to allow simulation to run for. Simulation is periodically checked and will halt if it
                        appears the next segment of the simulation will exceed max_wall_time. This is not guaranteed to
                        work since the guess is based on the run time up to that point.
                        Intended to be used when running on system with job manager.
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

    load_balance = LoadBalancer()
    ######################
    # DOMAIN/GEOMETRY/MESH
    ######################

    # Dimensions
    X_MAX = +channel_width / 2.
    X_MIN = -X_MAX
    Y_MAX = +channel_width / 2.
    Y_MIN = -Y_MAX
    Z_MAX = gap_distance
    Z_MIN = 0.

    # TODO: cells in all dimensions reduced by 10x for testing, will need to verify if this is reasonable (TEMP)
    # Grid parameters
    dx_want = 5e-9
    dy_want = 5e-9
    dz_want = 5e-9

    NUM_X = int(round(channel_width / dx_want))  # 20 #128 #10
    NUM_Y = int(round(channel_width / dy_want))  # 20 #128 #10
    NUM_Z = int(round(gap_distance / dz_want))

    # mesh spacing
    dz = (Z_MAX - Z_MIN) / NUM_Z
    dx = channel_width / NUM_X
    dy = channel_width / NUM_Y

    print "Channel width: {}, DX = {}".format(channel_width, dx)
    print "Channel width: {}, DY = {}".format(channel_width, dy)
    print "Channel length: {}, DZ = {}".format(gap_distance, dz)

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
    collector_voltage = phi_em - phi_coll

    # Emitted species
    background_beam = Species(type=Electron, name='background')
    measurement_beam = Species(type=Electron, name='measurement')

    # Emitter area and position
    SOURCE_RADIUS_1 = 0.5 * channel_width  # a0 parameter - X plane
    SOURCE_RADIUS_2 = 0.5 * channel_width  # b0 parameter - Y plane
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

    PTCL_PER_STEP = 100

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
        background_beam.b0 = SOURCE_RADIUS_2
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

        w3d.l_inj_exact = True

        # Specify thermal properties
        background_beam.vthz = measurement_beam.vthz = np.sqrt(EMITTER_TEMP * kb_J / background_beam.mass)
        background_beam.vthperp = measurement_beam.vthperp = np.sqrt(EMITTER_TEMP * kb_J / background_beam.mass)
        top.lhalfmaxwellinject = 1  # inject z velocities as half Maxwellian

        beam_current = sources.j_rd(EMITTER_TEMP, EMITTER_PHI) * cathode_area  # steady state current in Amps
        print('beam current expected: {}, current density {}'.format(beam_current, beam_current / cathode_area))
        jcl = 4. / 9. * eps0 * sqrt(2. * echarge / background_beam.mass) \
                       * ACCEL_VOLTS ** 1.5 / gap_distance ** 2 * cathode_area
        print('child-langmuir  limit: {}, current density {}'.format(jcl, jcl / cathode_area))
        background_beam.ibeam = measurement_beam.ibeam = beam_current
        background_beam.a0 = measurement_beam.a0 = SOURCE_RADIUS_1
        background_beam.b0 = measurement_beam.b0 = SOURCE_RADIUS_2
        background_beam.ap0 = measurement_beam.ap0 = .0e0
        background_beam.bp0 = measurement_beam.bp0 = .0e0

    derivqty()

    ##############
    # FIELD SOLVER
    ##############
    # Add Uniform B_z field if turned on
    if magnetic_field:
        bz = np.zeros([w3d.nx, w3d.ny, w3d.nz])
        bz[:, :, :] = magnetic_field
        z_start = w3d.zmmin
        z_stop = w3d.zmmax
        top.ibpush = 2
        addnewbgrd(z_start, z_stop, xs=w3d.xmmin, dx=(w3d.xmmax - w3d.xmmin), ys=w3d.ymmin, dy=(w3d.ymmax - w3d.ymmin),
                   nx=w3d.nx, ny=w3d.ny, nz=w3d.nz, bz=bz)

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
                                     channel_width)
        accel_grid.voltage = V_grid

    # --- Anode Location
    zplate = Z_MAX

    # Create source conductors
    if install_grid:
        source = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0., condid=2)
    else:
        source = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0.)

    # Create ground plate
    total_rho = efficiency.tec_parameters['rho_load'][0]
    if install_grid:
        plate = ZPlane(zcent=zplate, condid=3)
        circuit = ExternalCircuit(top, solverE, total_rho, collector_voltage, cathode_area * 1e4, plate, debug=False)
        plate.voltage = circuit
        # plate.voltage = collector_voltage
    else:
        plate = ZPlane(zcent=zplate)
        circuit = ExternalCircuit(top, solverE, total_rho, collector_voltage, cathode_area * 1e4, plate, debug=False)
        plate.voltage = circuit
        # plate.voltage = collector_voltage

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
        particleperiod = 250  # TEMP
        particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=top, w3d=w3d,
                                                   species={species.name: species
                                                            for species in listofallspecies},
                                                            # if species.name == 'measurement'}, # TEMP
                                                   comm_world=comm_world, lparallel_output=False,
                                                   write_dir=diagDir[:-5])
        installafterstep(particle_diagnostic_0.write)

    if field_diagnostic_switch:
        fieldperiod = 1000
        efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d,
                                                                  comm_world=comm_world,
                                                                  period=fieldperiod)
        installafterstep(efield_diagnostic_0.write)

    # Set externally derived parameters for efficiency calculation
    efficiency.tec_parameters['A_em'][0] = cathode_area * 1e4  # cm**2
    if install_grid:
        efficiency.tec_parameters['occlusion'][0] = efficiency.calculate_occlusion(**efficiency.tec_parameters)
    else:
        efficiency.tec_parameters['occlusion'][0] = 0.0

    ##########################
    # SOLVER SETTINGS/GENERATE
    ##########################

    # prevent gist from starting upon setup
    top.lprntpara = false
    top.lpsplots = false

    top.verbosity = -1  # Reduce solver verbosity
    solverE.mgverbose = -1  # further reduce output upon stepping - prevents websocket timeouts in Jupyter notebook

    init_iters = 20000
    regular_iters = 200

    init_tol = 1e-6
    regular_tol = 1e-6

    # Time Step

    # Determine an appropriate time step based upon estimated final velocity
    if install_grid:
        vz_accel = sqrt(2. * abs(V_grid) * np.abs(background_beam.charge) / background_beam.mass)
    else:
        vz_accel = sqrt(2. * abs(collector_voltage) * np.abs(background_beam.charge) / background_beam.mass)
    vzfinal = vz_accel + beam_beta * c
    dt = dz / vzfinal
    top.dt = dt

    solverE.mgmaxiters = init_iters
    solverE.mgtol = init_tol
    package("w3d")
    generate()
    solverE.mgtol = regular_tol
    solverE.mgmaxiters = regular_iters

    print("weights (background) (measurement): {}, {}".format(background_beam.sw, measurement_beam.sw))

    # Use rnpinject to set number of macroparticles emitted
    background_beam.rnpinject = PTCL_PER_STEP
    measurement_beam.rnpinject = 0  # measurement beam is off at start

    ##################
    # CONTROL SEQUENCE
    ##################
    # Run until steady state is achieved (flat current profile at collector) (measurement species turned on)
    # Record data for effiency calculation
    # Switch off measurement species and wait for simulation to clear (background species is switched on)

    early_abort = 0  # If true will flag output data to notify
    startup_time = 4 * gap_distance / vz_accel  # ~4 crossing times to approach steady-state with external circuit
    crossing_measurements = 10  # Number of crossing times to record for
    steps_per_crossing = int(gap_distance / vz_accel / dt)
    ss_check_interval = int(steps_per_crossing / 2.)
    ss_max_checks = 8  # Maximum number of of times to run steady-state check procedure before aborting
    times = []  # Write out timing of cycle steps to file
    clock = 0  # clock tracks the current, total simulation-runtime

    # Run initial block of steps
    record_time(stept, times, startup_time)
    clock += times[-1]
    stop_initialization = top.it  # for diag file

    print("Completed Initialization on Step {}\nInitialization run time: {}".format(top.it, times[-1]))

    # Start checking for Steady State Operation
    tol = 0.01
    ss_flag = 0
    check_count = 0  # Track number of times steady-state check performed

    while ss_flag != 1 and check_count < ss_max_checks:
        if (max_wall_time - clock) < times[-1]:
            early_abort = 1
            break

        record_time(step, times, ss_check_interval*4)
        clock += times[-1]

        tstart = (top.it - ss_check_interval) * top.dt
        _, current1 = plate.get_current_history(js=None, l_lost=1, l_emit=0,
                                               l_image=0, tmin=tstart, tmax=None, nt=1)
        current = np.sum(current1)

        if np.abs(current) < 0.5 * efficiency.tec_parameters['occlusion'][0] * beam_current:
            # If too little current is getting through run another check cycle
            check_count += 1
            print("Completed check {}, insufficient current, running again for {} steps".format(check_count,
                                                                                                ss_check_interval))
            continue

        ss_flag = 1
        # print np.abs(current), 0.5 * efficiency.tec_parameters['occlusion'][0] * beam_current
        # try:
        #     # If steady_state check initialized no need to do it again
        #     steady_state
        # except NameError:
        #     # If this is the first pass with sufficient current then initialize the check
        #     if check_count == 0:
        #         # If the initial period was long enough to get current on collector then use that
        #         steady_state = SteadyState(top, plate, steps_per_crossing)
        #     else:
        #         # If we had to run several steady state checks with no current then just use the period with current
        #         steady_state = SteadyState(top, plate, ss_check_interval)
        #
        # ss_flag = steady_state(steps_per_crossing)
        check_count += 1

    stop_ss_check = top.it  # For diag file

    # If there was a failure to reach steady state after specified number of checks then pass directly end
    if check_count == ss_max_checks:
        early_abort = -1
        crossing_measurements = 0
        print("Failed to reach steady state. Aborting simulation.")
    else:
        # Start Steady State Operation
        print(" Steady State Reached.\nStarting efficiency "
              "recording for {} crossing times.\nThis will be {} steps".format(crossing_measurements,
                                                                               steps_per_crossing * crossing_measurements))

    # particle_diagnostic_0.period = steps_per_crossing #TEMP commented out
    # Switch to measurement beam species
    measurement_beam.rnpinject = PTCL_PER_STEP
    background_beam.rnpinject = 0

    # Install Zcrossing Diagnostic
    ZCross = ZCrossingParticles(zz=grid_height * gap_distance / 200., laccumulate=1)
    emitter_flux = []

    crossing_wall_time = times[-1] * steps_per_crossing / ss_check_interval  # Estimate wall time for one crossing
    print('crossing_wall_time estimate: {}, for {} steps'.format(crossing_wall_time, steps_per_crossing))
    print('wind-down loop time estimate: {}, for {} steps'.format(crossing_wall_time *
                                                                  steps_per_crossing / ss_check_interval,
                                                                  ss_check_interval))
    for sint in range(crossing_measurements):
        # Kill the loop and proceed to writeout if we don't have time to complete the loop
        if (max_wall_time - clock) < crossing_wall_time:
            early_abort = 2
            break

        record_time(step, times, steps_per_crossing)
        clock += times[-1]

        # Re-evaluate time for next loop
        crossing_wall_time = times[-1]

        # Record velocities of emitted particles for later KE calculation
        velocity_array = np.array([ZCross.getvx(js=measurement_beam.js),
                                   ZCross.getvy(js=measurement_beam.js),
                                   ZCross.getvz(js=measurement_beam.js)]).transpose()
        # velocity_array = velocity_array[velocity_array[:, 2] >= 0.]  # Filter particles moving to emitter
        emitter_flux.append(velocity_array)

        ZCross.clear()  # Clear ZcrossingParticles memory

        print("Measurement: {} of {} intervals completed. Interval run time: {} s".format(sint + 1,
                                                                                          crossing_measurements,
                                                                                          times[-1]))
    stop_eff_calc = top.it  # For diag file
    # Run wind-down until measurement particles have cleared
    measurement_beam.rnpinject = 0
    background_beam.rnpinject = PTCL_PER_STEP

    initial_population = measurement_beam.npsim[0]
    measurement_tol = 0.03
    # if particle_diagnostic_switch:
    #     particle_diagnostic_0.period = ss_check_interval
    while measurement_beam.npsim[0] > measurement_tol * initial_population:
        # Kill the loop and proceed to writeout if we don't have time to complete the loop
        if (max_wall_time - clock) < crossing_wall_time * ss_check_interval / steps_per_crossing :
            early_abort = 3
            break

        record_time(step, times, ss_check_interval)
        clock += times[-1]

        # Record velocities of emitted particles for later KE calculation
        # Check is required here as measurement_beam particles will not always be passing through
        if ZCross.getvx(js=measurement_beam.js).shape[0] > 0:
            velocity_array = np.array([ZCross.getvx(js=measurement_beam.js),
                                       ZCross.getvy(js=measurement_beam.js),
                                       ZCross.getvz(js=measurement_beam.js)]).transpose()
            print "Backwards particles: {}".format(np.where(velocity_array[:, 2] < 0.)[0].shape[0])
            # velocity_array = velocity_array[velocity_array[:, 2] >= 0.]  # Filter particles moving to emitter
            emitter_flux.append(velocity_array)
            ZCross.clear()  # Clear ZcrossingParticles memory
        print(" Wind-down: Taking {} steps, On Step: {}, {} Particles Left".format(ss_check_interval, top.it,
                                                                                   measurement_beam.npsim[0]))

    stop_winddown = top.it  # For diag file

    ######################
    # CALCULATE EFFICIENCY
    ######################
    try:
        emitter_flux = np.vstack(emitter_flux)
    except ValueError:
        # If this triggered then measurement emission never took place
        # Run took too long probably and abort took place
        emitter_flux = np.array([[0., 0., 0.]])

    # Find integrated charge on each conductor
    surface_charge = analyze_scraped_particles(top, measurement_beam, solverE)
    measured_charge = {}

    for key in surface_charge:
        # We can abuse the fact that js=0 for background species to filter it from the sum
        measured_charge[key] = np.sum(surface_charge[key][:, 1] * surface_charge[key][:, 3])

    # Set derived parameters from simulation
    efficiency.tec_parameters['run_time'][0] = crossing_measurements * steps_per_crossing * dt
    if crossing_measurements == 0:
        # Set to large value to force all powers and currents to zero
        efficiency.tec_parameters['run_time'][0] = 1e20

    # Find total number of measurement particles that were emitted
    total_macroparticles = measurement_beam.npsim[0] + np.sum([measured_charge[key] for key in surface_charge])
    efficiency.tec_parameters['J_em'][0] = e * (total_macroparticles - measured_charge[scraper_dictionary['source']]) \
                                        * measurement_beam.sw / \
                                        efficiency.tec_parameters['run_time'][0] / efficiency.tec_parameters['A_em'][0]

    # If grid isn't being used then J_grid will not be in scraper dict
    try:
        efficiency.tec_parameters['J_grid'][0] = e * measured_charge[scraper_dictionary['grid']] * measurement_beam.sw / \
                                            efficiency.tec_parameters['run_time'][0] / \
                                            (efficiency.tec_parameters['occlusion'][0] *
                                             efficiency.tec_parameters['A_em'][0])
    except KeyError:
        efficiency.tec_parameters['J_grid'][0] = 0.0

    efficiency.tec_parameters['J_ec'][0] = e * measured_charge[scraper_dictionary['collector']] * measurement_beam.sw / \
                                        efficiency.tec_parameters['run_time'][0] / efficiency.tec_parameters['A_em'][0]

    efficiency.tec_parameters['P_em'][0] = efficiency.calculate_power_flux(emitter_flux, measurement_beam.sw,
                                                                        efficiency.tec_parameters['phi_em'][0],
                                                                        **efficiency.tec_parameters)

    # Efficiency calculation
    print("Efficiency")
    efficiency_result = efficiency.calculate_efficiency(**efficiency.tec_parameters)
    print("Overall Efficiency: {}".format(efficiency_result['eta']))
    print("Total steps: {}".format(top.it))
    ######################
    # FINAL RUN STATISTICS
    ######################

    if comm_world.rank == 0:
        if not os.path.exists('diags_id{}'.format(run_id)):
            os.makedirs('diags_id{}'.format(run_id))

        np.save('iv_data.npy', np.array([circuit.current_history, circuit.voltage_history]))

        write_parameter_file(run_attributes, filename='diags_id{}/'.format(run_id))

        filename = 'efficiency_id{}.h5'.format(str(run_id))
        with h5.File(os.path.join('diags_id{}'.format(run_id), filename), 'w') as h5file:
            # TODO: Add current history
            eff_group = h5file.create_group('/efficiency')
            run_group = h5file.create_group('/attributes')
            scrap_group = h5file.create_group('/scraper')
            h5file.attrs['complete'] = early_abort
            for key in efficiency_result:
                eff_group.attrs[key] = efficiency_result[key]
            for key in efficiency.tec_parameters:
                eff_group.attrs[key] = efficiency.tec_parameters[key]
            for key in run_attributes:
                run_group.attrs[key] = run_attributes[key]
            run_group.attrs['dt'] = top.dt
            run_group.attrs['stop_initialization'] = stop_initialization
            run_group.attrs['stop_ss_check'] = stop_ss_check
            run_group.attrs['stop_eff_calc'] = stop_eff_calc
            run_group.attrs['stop_winddown'] = stop_winddown
            # for key, value in scraper_dictionary.iteritems():
            #     scrap_group.attrs[key] = measured_charge[value]
            #
            inv_scraper_dict = {value: key for key, value in scraper_dictionary.iteritems()}
            for cond in solverE.conductordatalist:
                cond_objs = cond[0]
                scrap_group.attrs[inv_scraper_dict[cond_objs.condid]] = measured_charge[cond_objs.condid]
                _, bckgrnd_current = cond_objs.get_current_history(js=0, l_lost=1, l_emit=0,
                                                                   l_image=0, tmin=None, tmax=None, nt=top.it)
                _, msrmnt_current = cond_objs.get_current_history(js=1, l_lost=1, l_emit=0,
                                                                  l_image=0, tmin=None, tmax=None, nt=top.it)
                scrap_group.create_dataset('{}_background'.format(inv_scraper_dict[cond_objs.condid]),
                                           data=bckgrnd_current)
                scrap_group.create_dataset('{}_measurement'.format(inv_scraper_dict[cond_objs.condid]),
                                           data=msrmnt_current)

            h5file.create_dataset('times', data=times)


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
