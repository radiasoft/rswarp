from __future__ import division


import h5py as h5

# set warpoptions.ignoreUnknownArgs = True before main import to allow command line arguments
import warpoptions
warpoptions.ignoreUnknownArgs = True

from warp import *

from copy import deepcopy
from rswarp.cathode import sources
from warp.data_dumping.openpmd_diag import ParticleDiagnostic

# This path is hardcoded until issues with the stlconductor center modification are resolved
sys.path.insert(2, '/home/vagrant/jupyter/repos/rswarp/rswarp/')

from diagnostics import FieldDiagnostic
from utilities.file_utils import cleanupPrevious
from diagnostics.parallel import save_lost_particles
from diagnostics.ConductorDiagnostics import analyze_scraped_particles
from stlconductor.stlconductor import *

# path to particle collector
path_to_pr = r"."
if not path_to_pr in sys.path: sys.path.insert(2, path_to_pr)
from particlereflector import *

# grid scraper
from gridscraper import ParticleScraperGrid


if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=True, error=False)

# Constants imports
from scipy.constants import e, m_e, c, k

kb_eV = 8.6173324e-5  # Bolztmann constant in eV/K
kb_J = k  # Boltzmann constant in J/K

m = m_e  # electron mass

num_particles_res = 0


def main(injection_type, cathode_temperature, cathode_workfunction, anode_workfunction, anode_voltage,
         gate_voltage, lambdaR, beta, srefprob, drefprob, reflection_scheme,
         gap_voltage, dgap, dt, nsteps, particles_per_step, file_path,
         fieldperiod=100, particleperiod=1000, reflections=True):
    settings = deepcopy(locals())
    ############################
    # Domain / Geometry / Mesh #
    ############################
    
    
    # Grid geometry
    
    w = 1.6e-3  # full hexagon width
    a = w / 2.0  # inner wall width
    b = 80e-6  # thickness
    r = np.sqrt(3) / 2. * w  # distance between two opposite walls
    d = np.sqrt(b**2 / 4. + b**2)  # 
    h = 0.2e-3  # height of grid (length in z)
    
    PLATE_SPACING = dgap
    CHANNEL_WIDTH_X = w + 2 * d + a
    CHANNEL_WIDTH_Y = 2 * r + 2 * b
    
    # Dimensions
    X_MAX = +CHANNEL_WIDTH_X / 2.
    X_MIN = -X_MAX
    Y_MAX = +CHANNEL_WIDTH_Y / 2.
    Y_MIN = -Y_MAX
    Z_MAX = PLATE_SPACING
    Z_MIN = 0.

    # Grid parameters
    NUM_X = 100; NUM_Y = 100; NUM_Z = 25

    # z step size
    dx = (X_MAX - X_MIN)/NUM_X
    dy = (Y_MAX - Y_MIN)/NUM_Y
    dz = (Z_MAX - Z_MIN)/NUM_Z

    print(" --- (xmin, ymin, zmin) = ({}, {}, {})".format(X_MIN, Y_MIN, Z_MIN))
    print(" --- (xmax, ymax, zmax) = ({}, {}, {})".format(X_MAX, Y_MAX, Z_MAX))
    print(" --- (dx, dy, dz) = ({}, {}, {})".format(dx, dy, dz))

    # Solver Geometry and Boundaries

    # Specify solver geometry
    w3d.solvergeom = w3d.XYZgeom

    # Set field boundary conditions
    w3d.bound0 = dirichlet
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
    w3d.zmmin = Z_MIN
    w3d.zmmax = Z_MAX

    # Set mesh cell counts
    w3d.nx = NUM_X
    w3d.ny = NUM_Y
    w3d.nz = NUM_Z

    ################
    # FIELD SOLVER #
    ################
    magnetic_field = True
    if magnetic_field:
        bz = np.zeros([w3d.nx, w3d.ny, w3d.nz])
        bz[:, :, :] = 200e-3
        z_start = w3d.zmmin
        z_stop = w3d.zmmax
        top.ibpush = 2
        addnewbgrd(z_start, z_stop, xs=w3d.xmmin, dx=(w3d.xmmax - w3d.xmmin), ys=w3d.ymmin, dy=(w3d.ymmax - w3d.ymmin),
                    nx=w3d.nx, ny=w3d.ny, nz=w3d.nz, bz=bz)

    # Set up fieldsolver
    f3d.mgtol = 1e-6
    solverE = MultiGrid3D()
    registersolver(solverE)

    ###############################
    # PARTICLE INJECTION SETTINGS #
    ###############################
    volts_on_conductor = gap_voltage

    # INJECTION SPECIFICATION
    USER_INJECT = injection_type

    # Cathode and anode settings
    CATHODE_TEMP = cathode_temperature
    CATHODE_PHI = cathode_workfunction
    ANODE_WF = anode_workfunction            # Can be used if vacuum level is being set
    CONDUCTOR_VOLTS = volts_on_conductor     # ACCEL_VOLTS used for velocity and CL calculations

    # Emitted species
    background_beam = Species(type=Electron, name='background')
    # Reflected species
    reflected_electrons = Species(type=Electron, name='reflected')

    # Emitter area and position
    SOURCE_RADIUS_1 = 0.5 * CHANNEL_WIDTH_X  # a0 parameter - X plane
    SOURCE_RADIUS_2 = 0.5 * CHANNEL_WIDTH_Y  # b0 parameter - Y plane
    Z_PART_MIN = dz / 1000.                  # starting particle z value

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

    PTCL_PER_STEP = particles_per_step
    if USER_INJECT == 1:
        CURRENT_MODIFIER = 0.5  # Factor to multiply CL current by when setting beam current
        # Constant current density - beam transverse velocity fixed to zero, very small longitduinal velocity

        # Set injection flag
        top.inject = 1  # 1 means constant; 2 means space-charge limited injection;# 6 means user-specified
        top.npinject = PTCL_PER_STEP
        beam_current = 4. / 9. * eps0 * sqrt(2. * echarge / background_beam.mass) \
                           * CONDUCTOR_VOLTS ** 1.5 / PLATE_SPACING ** 2 * cathode_area

        background_beam.ibeam = beam_current * CURRENT_MODIFIER

        background_beam.a0 = SOURCE_RADIUS_1
        background_beam.b0 = SOURCE_RADIUS_2
        background_beam.ap0 = .0e0
        background_beam.bp0 = .0e0

        w3d.l_inj_exact = True

        # Initial velocity settings (5% of c)
        vrms = np.sqrt(1 - 1 / (0.05 / 511e3 + 1) ** 2) * 3e8
        top.vzinject = vrms

    if USER_INJECT == 2:
        # SC limited Thermionic injection
        top.inject = 2

        # Set both beams to same npinject to keep weights the same
        background_beam.npinject = PTCL_PER_STEP
        top.finject = [1.0, 0.0]
        w3d.l_inj_exact = True

        # Specify thermal properties
        background_beam.vthz = np.sqrt(CATHODE_TEMP * kb_J / background_beam.mass)
        background_beam.vthperp =  np.sqrt(CATHODE_TEMP * kb_J / background_beam.mass)
        top.lhalfmaxwellinject = 1  # inject z velocities as half Maxwellian

        beam_current = sources.j_rd(CATHODE_TEMP, CATHODE_PHI) * cathode_area  # steady state current in Amps
        print('beam current expected: {}, current density {}'.format(beam_current, beam_current / cathode_area))
        jcl = 0.
        if gap_voltage > 0.:
            jcl = 4. / 9. * eps0 * sqrt(2. * echarge / background_beam.mass) \
                * CONDUCTOR_VOLTS ** 1.5 / PLATE_SPACING ** 2 * cathode_area
        print('child-langmuir  limit: {}, current density {}'.format(jcl, jcl / cathode_area))
        background_beam.ibeam  = beam_current
        background_beam.a0  = SOURCE_RADIUS_1
        background_beam.b0  = SOURCE_RADIUS_2
        background_beam.ap0  = .0e0
        background_beam.bp0  = .0e0

    if USER_INJECT == 4:
        w3d.l_inj_exact = True
        w3d.l_inj_user_particles_v = True

        # Schottky model
        top.inject = 1
        top.ninject = 1
        top.lhalfmaxwellinject = 1  # inject z velocities as half Maxwellian
        top.zinject = np.asarray([Z_PART_MIN])
        top.ainject = np.asarray([SOURCE_RADIUS_1])
        top.binject = np.asarray([SOURCE_RADIUS_2])
        top.finject = np.asarray([[1.0, 0.0]])

        electric_field = 0
        delta_w = np.sqrt(e ** 3 * electric_field / (4 * np.pi * eps0))
        A0 = 1.20173e6
        AR = A0*lambdaR

        background_beam.a0 = SOURCE_RADIUS_1
        background_beam.b0 = SOURCE_RADIUS_2
        background_beam.ap0 = .0e0
        background_beam.bp0 = .0e0
        background_beam.vthz = np.sqrt(CATHODE_TEMP * kb_J / background_beam.mass)
        background_beam.vthperp = np.sqrt(CATHODE_TEMP * kb_J / background_beam.mass)

        # use Richardson current to estimate particle weight
        rd_current = AR * CATHODE_TEMP ** 2 * np.exp(- (CATHODE_PHI * e) / (CATHODE_TEMP * k)) * cathode_area
        electrons_per_second = rd_current / e
        electrons_per_step = electrons_per_second * dt
        background_beam.sw = electrons_per_step / PTCL_PER_STEP

        def schottky_emission():
            # schottky emission at cathode side
            
            global num_particles_res
            
            Ez = solverE.getez()
            Ez_mean = np.mean(Ez[:, :, 0])

            if w3d.inj_js == background_beam.js:
                delta_w = 0.
                if Ez_mean < 0.:
                    delta_w = np.sqrt(beta * e ** 3 * np.abs(Ez_mean) / (4 * np.pi * eps0))

                rd_current = AR * CATHODE_TEMP ** 2 * np.exp(
                    - (CATHODE_PHI * e - delta_w) / (CATHODE_TEMP * k)) * cathode_area
                electrons_per_second = rd_current / e
                electrons_per_step = electrons_per_second * top.dt
                float_num_particles = electrons_per_step / background_beam.sw
                num_particles = int(float_num_particles + num_particles_res + np.random.rand())
                num_particles_res += float_num_particles - num_particles
                
                # --- inject np particles of species electrons1
                # --- Create the particles on the surface
                x = -background_beam.a0 + 2 * background_beam.a0 * np.random.rand(num_particles)
                y = -background_beam.b0 + 2 * background_beam.b0 * np.random.rand(num_particles)
                vz = np.random.rand(num_particles)
                vz = np.maximum(1e-14 * np.ones_like(vz), vz);
                vz = background_beam.vthz * np.sqrt(-2.0 * np.log(vz))

                vrf = np.random.rand(num_particles)
                vrf = np.maximum(1e-14 * np.ones_like(vrf), vrf)
                vrf = background_beam.vthz * np.sqrt(-2.0 * np.log(vrf))
                trf = 2 * np.pi * np.random.rand(num_particles)
                vx = vrf * np.cos(trf)
                vy = vrf * np.sin(trf)

                # --- Setup the injection arrays
                w3d.npgrp = num_particles
                gchange('Setpwork3d')
                # --- Fill in the data. All have the same z-velocity, vz1.
                w3d.xt[:] = x
                w3d.yt[:] = y
                w3d.uxt[:] = vx
                w3d.uyt[:] = vy
                w3d.uzt[:] = vz

        installuserparticlesinjection(schottky_emission)

    derivqty()

    print("weight:", background_beam.sw)

    ##########################
    # CONDUCTOR INSTALLATION #
    ##########################
    install_conductor = True
    
    # --- Anode Location
    zplate = Z_MAX

    # Create source conductors
    if install_conductor:
        emitter = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0., condid=2)
    else:
        emitter = ZPlane(zcent=w3d.zmmin, zsign=-1., voltage=0.)

    # Create collector
    if install_conductor:
        collector = ZPlane(voltage=gap_voltage, zcent=zplate, condid=3)
    else:
        collector = ZPlane(voltage=gap_voltage, zcent=zplate)
        
    # Create grid
    
    if install_conductor:
        # Offsets to make sure grid is centered after install
        # Because case089 was not made with a hexagon at (0., 0.) it must be moved depending on STL file used and
        # version of STLconductor
        cxmin,cymin,czmin = -0.00260785012506,-0.00312974047847,0.000135000009323 
        cxmax,cymax,czmax = 0.00339215015993,0.00287025980651,0.000335000018822
        conductor = STLconductor("honeycomb_case0.89t_xycenter_zcen470.stl",
                                 xcent=cxmin + (cxmax - cxmin) / 2., ycent=cymin + (cymax - cymin) / 2., #zcent=czmin + (czmax - czmin) / 2., disp=(0.,0.,1e-6),
                                 verbose="on", voltage=gate_voltage, normalization_factor=dz, condid=1)

    if install_conductor :
        installconductor(conductor, dfill=largepos)
        installconductor(emitter, dfill=largepos)
        installconductor(collector, dfill=largepos)
        scraper_diode = ParticleScraper([emitter, collector],
                                  lcollectlpdata=True,
                                  lsaveintercept=True)
        scraper_gate  = ParticleScraper([conductor],
                                  lcollectlpdata=True,
                                  lsaveintercept=False)
        
        scraper_dictionary = {1: 'grid', 2: 'emitter', 3: 'collector'}
    else:
        installconductor(emitter, dfill=largepos)
        installconductor(collector, dfill=largepos)
        scraper = ParticleScraper([emitter, collector],
                                  lcollectlpdata=True,
                                  lsaveintercept=True)
        scraper_dictionary = {1: 'emitter', 2: 'collector'}
        
    ########################
    # Hacked Grid Scraping #
    ########################
    # Implements boxes with a reflection probability based on transparency attribute
    # used to crudely emulate particles reflected from the STLconductor honeycomb which cannot provide
    # scraper positions needed for reflection calculation
    grid_reflections = True
    
    if grid_reflections:
        grid_front = Box(xcent=0., ycent=0., zcent=0.135e-3 - dz / 2., xsize=2 * (X_MAX- X_MIN), ysize=2 *(Y_MAX - Y_MIN), zsize= dz)
        grid_back = Box(xcent=0., ycent=0., zcent=0.335e-3 + dz / 2, xsize=2 * (X_MAX- X_MIN), ysize=2 *(Y_MAX - Y_MIN), zsize= dz)

        scraper_front = ParticleScraperGrid(grid_front,
                                      lcollectlpdata=True,
                                      lsaveintercept=True)
        scraper_back = ParticleScraperGrid(grid_back,
                                      lcollectlpdata=True,
                                      lsaveintercept=True)

        # Fraction of particles expected to pass through the conductor (assuming they cross it in a single step)
        scraper_front.transparency = 0.80
        scraper_back.transparency = 0.80

        # Directional scraper to prevent particles from being scraped coming out of honeycomb interior
        scraper_front.directional_scraper = -1
        scraper_back.directional_scraper = +1
    
    #####################
    # Diagnostics Setup #
    #####################

    efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d,
                                                              comm_world=comm_world,
                                                              period=fieldperiod, write_dir=os.path.join(file_path,'fields'))
    installafterstep(efield_diagnostic_0.write)

    particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=top, w3d=w3d,
                                               species={species.name: species for species in listofallspecies},
                                               comm_world=comm_world, lparallel_output=False, write_dir=file_path)
    installafterstep(particle_diagnostic_0.write)

    ####################
    # CONTROL SEQUENCE #
    ####################

    # prevent gist from starting upon setup
    top.lprntpara = false
    top.lpsplots = false

    top.verbosity = 1      # Reduce solver verbosity
    solverE.mgverbose = 1  # further reduce output upon stepping - prevents websocket timeouts in Jupyter notebook

    init_iters = 2000
    regular_iters = 50

    init_tol = 1e-5
    regular_tol = 1e-6

    # Time Step
    top.dt = dt

    # Define and install particle reflector
    if reflections:
        collector_reflector = ParticleReflector(scraper=scraper_diode, conductor=collector,
                                                spref=reflected_electrons,
                                                srefprob=srefprob, drefprob=drefprob,
                                                refscheme=reflection_scheme)
        installparticlereflector(collector_reflector)
        print("reflection_scheme = "+reflection_scheme)
    else:
        print("reflections: off")
        
    if grid_reflections:
        reflector_front = ParticleReflector(scraper=scraper_front, conductor=grid_front,
                                                spref=reflected_electrons,
                                                srefprob=srefprob, drefprob=0.75,
                                                refscheme=reflection_scheme)
        installparticlereflector(reflector_front)
        
        reflector_back = ParticleReflector(scraper=scraper_back, conductor=grid_back,
                                                spref=reflected_electrons,
                                                srefprob=srefprob, drefprob=0.75,
                                                refscheme=reflection_scheme)
        installparticlereflector(reflector_back)


    # initialize field solver and potential field
    solverE.mgmaxiters = init_iters
    solverE.mgtol = init_tol

    package("w3d")
    generate()

    # Specify particle weight for reflected_electrons
    reflected_electrons.sw = background_beam.sw
    print("weight: background_beam = {}, reflected = {}".format(background_beam.sw, reflected_electrons.sw))

    solverE.mgmaxiters = regular_iters
    solverE.mgtol = regular_tol
    step(nsteps)
    
    
    ####################
    # Final Output     #
    ####################

    surface_charge = analyze_collected_charge(top, solverE)
    if reflections:
        reflected_charge = analyze_reflected_charge(top, [collector_reflector], comm_world=comm_world)
            
    if comm_world.rank == 0:
        filename = os.path.join(file_path, "all_charge_anodeV_{}.h5".format(anode_voltage))
        diag_file = h5.File(filename, 'w')
        
        # Write simulation parameters
        for key, val in settings.items():
            diag_file.attrs[key] = val
        for dom_attr in ['xmmin', 'xmmax', 'ymmin', 'ymmax', 'zmmin', 'zmmax', 'nx', 'ny', 'nz']:
            diag_file.attrs[dom_attr] = eval('w3d.' + dom_attr)
        
        # Record scraped particles into scraper group of file
        scraper_data = diag_file.create_group('scraper')
        for key, val in scraper_dictionary.items():
            scraper_data.attrs[val] = key
        
        for condid, cond_data in surface_charge.items():
            cond_group = scraper_data.create_group('{}'.format(scraper_dictionary[condid]))
            for i, spec_dat in enumerate(cond_data):
                cond_group.create_dataset(listofallspecies[i].name, data=spec_dat)
        
        # Record reflected particles into reflector group
        if reflections:
            reflector_data = diag_file.create_group('reflector')
            for key in reflected_charge:
                reflector_data.attrs[scraper_dictionary[key]] = key
            
            for condid, ref_data in reflected_charge.items():
                refl_group = reflector_data.create_group('{}'.format(scraper_dictionary[condid]))
                refl_group.create_dataset('reflected', data=ref_data)
        
        diag_file.close()


if __name__ == "__main__":
    anode_voltage = 1.2
    file_path = 'grid_reflection_test1/'
    
    if len(sys.argv) > 1:
        anode_voltage = float(sys.argv[1])

    run_options = {
        'dgap': 0.47e-3,
        'dt': 8e-12,
        'nsteps': 4000,
        'particles_per_step': 1600,
        'injection_type': 2,  # Thermionic injection without field enhancement
        'lambdaR': 0.5,       # Richardson constant AR = A0*lambdaR
        'srefprob': 0.0,      # probability of specular reflection
        'drefprob': 0.5,      # probability of diffuse reflection
        'cathode_temperature': 1050 + 273.15,
        'cathode_workfunction': 2.22, # in eV
        'anode_workfunction': 2.22 - 0.4,
        'anode_voltage': anode_voltage,
        'gap_voltage': anode_voltage + 0.4,
        'gate_voltage': 5.0,
        'beta': 27.,
        'reflection_scheme': "uniform",
        'reflections': True,
        'fieldperiod': 500,
        'particleperiod': 250,
        'file_path': file_path
    }


    print("run simulation with: anode voltage = {}, gap voltage = {}, beta = {}, gap size = {}, dt = {}, nsteps = {}, particles_per_step = {}, reflection_scheme = {}".format(
        run_options['anode_voltage'], run_options['gap_voltage'], run_options['beta'], run_options['dgap'], run_options['dt'], run_options['nsteps'], run_options['particles_per_step'], run_options['reflection_scheme']))

    main(**run_options)
