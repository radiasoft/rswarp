from __future__ import division
import warp as wp
import numpy as np
import h5py as h5
import os

from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.particles.singleparticle import TraceParticle
from warp.particles.extpart import ZCrossingParticles

# Use more-up-to-date local rswarp
import sys
sys.path.insert(1, '/home/vagrant/jupyter/rswarp')
from rswarp.diagnostics import FieldDiagnostic
from rswarp.cathode.injectors import UserInjectors
from rswarp.utilities.file_utils import cleanupPrevious

import rsoopic.h2crosssections as h2crosssections
sys.path.insert(1, '/home/vagrant/jupyter/rswarp/rswarp/ionization')
from ionization import Ionization
import crosssections as Xsect

# Seed set for reproduction
np.random.seed(967644)

diagDir = 'diags/hdf5'
diagFDir = ['diags/fields/magnetic', 'diags/fields/electric']

# Cleanup command if directories already exist
# Warp's HDF5 diagnostics will not overwrite existing files. This command cleans the diagnostic directory
# to allow for rerunning the simulation
#if wp.comm_world.rank == 0:
#    cleanupPrevious(diagDir, diagFDir)
#    try:
#        os.remove('./diags/crossing_record.h5')
#    except OSError:
#        pass

####################
# General Parameters
####################
# Set the solver geometry to cylindrical
wp.w3d.solvergeom = wp.w3d.RZgeom

# Switches
particle_diagnostic_switch = True  # Record particle data periodically
field_diagnostic_switch = True  # Record field/potential data periodically
user_injection = True  # Switches injection type
space_charge = True  # Controls field solve on/off
simulateIonization = True  # Include ionization in simulation

if user_injection:
    # User injection thermionic_rz_injector method uses a r**2 scaling to distribute particles uniformly
    variable_weight = False
else:
    # Warp default injection uses radially weighted particles to make injection uniform in r
    variable_weight = True

# Dimensions for the electron cooler
pipe_radius = 0.1524 / 2.  # m (Based on ECE specs)
cooler_length = 2.0  # m

cathode_temperature = 0.25  # eV

# Beam
beam_beta = 0.990813945176
beam_ke = wp.emass / wp.jperev * wp.clight**2 * (1. / np.sqrt(1-beam_beta**2) - 1.)  # eV
#print '*** beam_ke, beam_gamma =', beam_ke,  1. / np.sqrt(1-beam_beta**2)
beam_current = 10e-3  # A
beam_radius = 0.01  # m

##########################
# 3D Simulation Parameters
##########################

# Set cells (nx == ny in cylindrical coordinates)
wp.w3d.nx = 128
wp.w3d.ny = 128
wp.w3d.nz = 1024

# Set field boundary conditions (Warp only allows setting x and y together)
wp.w3d.bound0 = wp.neumann  # Use neumann to remove field from electrode edge at boundary
wp.w3d.boundnz = wp.neumann
wp.w3d.boundxy = wp.neumann

# Set boundary dimensions
wp.w3d.xmmin = -pipe_radius
wp.w3d.xmmax = pipe_radius
wp.w3d.ymmin = -pipe_radius
wp.w3d.ymmax = pipe_radius
wp.w3d.zmmin = 0.0
wp.w3d.zmmax = cooler_length

# Set particle boundary conditions
wp.top.pbound0 = wp.absorb
wp.top.pboundnz = wp.absorb

# Type of particle push. 2 includes tan correction to Boris push.
wp.top.ibpush = 2

dz = (wp.w3d.zmmax - wp.w3d.zmmin) / wp.w3d.nz  # Because warp doesn't set w3d.dz until after solver instantiated
wp.top.dt = 0.75 * dz / (beam_beta * wp.clight)  # Timestep set to cell crossing time with some padding

####################################
# Create Beam and Set its Parameters
####################################
# parameters for electron beam:
#   total energy: 3.2676813658 MeV
#   beta_e: 0.990813945176

wp.top.lrelativ = True
wp.top.relativity = True

ptcl_per_step = 1000  # number of particles to inject on each step
wp.top.npinject = ptcl_per_step

beam = wp.Species(type=wp.Electron, name='Electron', lvariableweights=variable_weight)
if space_charge:
    beam.ibeam = beam_current
else:
    beam.ibeam = 0.0

wp.w3d.l_inj_exact = True  # if true, position and angle of injected particle are
#  computed analytically rather than interpolated

"""
A custom injector routine is used here to allow for a very relativistic beam to be injected directly into the simulation
because Warp's built in routine is not based on relativistic kinematics. Setting user_injection = False should work
well for lower energy beams.
"""

if user_injection:
    wp.top.inject = 6  # Option 6 is a user specified input distribution
    wp.top.ainject = beam_radius  # Width of injection area in x
    wp.top.binject = beam_radius  # Wdith of injection area in y
    injector = UserInjectors(beam, wp.w3d, wp.gchange, cathode_temperature=cathode_temperature,
                             cathode_radius=beam_radius,
                             ptcl_per_step=ptcl_per_step, accelerating_voltage=beam_ke, zmin_scale=0.545)

    wp.installuserinjection(injector.thermionic_rz_injector)
else:
    wp.top.inject = 1  # Option 1 is a constant injection (default: particles injected from xy plane at z=0)
    wp.top.lhalfmaxwellinject = True  # Use half maxwell in axis perpendicular to emission surface (full in other axes)

    beam.a0 = beam_radius  # Width of injected beam in x
    beam.b0 = beam_radius  # Width of injected beam in y
    beam.ap0 = 0.0  # Width of injected beam in vx/vz
    beam.bp0 = 0.0  # Width of injected beam in vy/vz

    beam.vbeam = beam_beta * wp.clight
    assert beam.vbeam < 0.1 * wp.clight, "Injection velocity > 0.1c. " \
                                         "Constant injection does not use relativistic kinematics"
    beam.vthz = np.sqrt(cathode_temperature * wp.jperev / beam.mass)  # z thermal velocity
    beam.vthperp = np.sqrt(cathode_temperature * wp.jperev / beam.mass)  # x and y thermal velocity
    wp.w3d.l_inj_rz = (wp.w3d.solvergeom == wp.w3d.RZgeom)

##############################
# Ionization of background gas
##############################

# Ionization e- + H2 -> H2+ + e- + e-

if simulateIonization is True:
    # Particle species for emission products of an ionization event
    h2plus = wp.Species(type=wp.Dihydrogen, charge_state=+1, name='H2+', weight=1000)
    emittedelec = wp.Species(type=wp.Electron, name='emitted e-', weight=1000)

    target_pressure = 0.4  # in Pa
    target_temp = 273.0  # in K
    target_density = target_pressure / wp.boltzmann / target_temp  # in 1/m^3

    # Instantiate Ionization
    # Dimensions control where background gas reservoir will be present in the domain
    ioniz = Ionization(
        stride=100,  # Number of particles allowed to be involved in ionization events per timestep
        xmin=wp.w3d.xmmin,
        xmax=wp.w3d.xmmax,
        ymin=wp.w3d.ymmin,
        ymax=wp.w3d.ymmax,
        zmin=wp.w3d.zmmin,
        zmax=wp.w3d.zmmax,
        nx=wp.w3d.nx,
        ny=wp.w3d.ny,
        nz=wp.w3d.nz,
        l_verbose=True
    )

    # add method used to add possible ionization events
    h2xs = Xsect.H2IonizationEvent()
    def xswrapper(vi):
        return h2xs.getCrossSection(vi)
    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, emittedelec],  # iterable of species created from ionization
        #cross_section=h2crosssections.h2_ioniz_crosssection,  # Cross section, can be float or function
        cross_section=xswrapper,
        #emitted_energy0=[0, h2crosssections.ejectedEnergy],  # Energy of each emitted species, can be float or function
        # or set to 'thermal' to create ions with a thermal energy spread set by temperature
        emitted_energy0=['thermal', h2crosssections.ejectedEnergy],  # Energy of each emitted species, can be float or function
        emitted_energy_sigma=[0, 0],  # Energy spread of emitted species (gives width of gaussian distribution)
        temperature=[target_temp, None],
        sampleEmittedAngle=h2crosssections.generateAngle,
        writeAngleDataDir=False,  # Write file recording statistics of angles
        writeAnglePeriod=1000,  # Period to write angle data, if used
        l_remove_incident=False,  # Remove incident macroparticles involved in ionization
        l_remove_target=False,  # Remove target macroparticles (only used if target_species set)
        ndens=target_density  # Background gas density (if target is reservoir of gas and not macroparticles)
    )

####################
# Install Conductors
####################
# Create conductors that will represent electrodes placed inside vacuum vessel

pipe_voltage = 0.0  # Set main pipe section to ground
electrode_voltage = +2e3  # electrodes held at several kV relative to main pipe
assert electrode_voltage < beam_ke, "Electrodes potential greater than beam KE."

pipe_radius = pipe_radius
electrode_length = 0.25
electrode_gap = 0.1
pipe_length = cooler_length - 2 * electrode_length - 2 * electrode_gap

z_positions = [0.0]
z_positions.append(z_positions[-1] + electrode_length)
z_positions.append(z_positions[-1] + electrode_gap)
z_positions.append(z_positions[-1] + pipe_length)
z_positions.append(z_positions[-1] + electrode_gap)
z_positions.append(z_positions[-1] + electrode_length)

conductors = []

entrance_electrode = wp.ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                     zlower=z_positions[0], zupper=z_positions[1])
conductors.append(entrance_electrode)

beam_pipe = wp.ZCylinderOut(voltage=pipe_voltage, radius=pipe_radius,
                            zlower=z_positions[2], zupper=z_positions[3])
conductors.append(beam_pipe)

exit_electrode = wp.ZCylinderOut(voltage=electrode_voltage, radius=pipe_radius,
                                 zlower=z_positions[4], zupper=z_positions[5])
conductors.append(exit_electrode)

##############################
# Install Ideal Solenoid Field
##############################
# electron cooler interaction takes place inside a solenoid
# for time being this is represented as an ideal magnetic field B = (0, 0, bz)

# External B-Field can be added by setting vector components at each cell
bz = np.zeros([wp.w3d.nx, wp.w3d.ny, wp.w3d.nz])
bz[:, :, :] = 1.0  # T
z_start = wp.w3d.zmmin
z_stop = wp.w3d.zmmax

# Add B-Field to simulation
wp.addnewbgrd(z_start, z_stop,
              xs=wp.w3d.xmmin, dx=(wp.w3d.xmmax - wp.w3d.xmmin),
              ys=wp.w3d.ymmin, dy=(wp.w3d.ymmax - wp.w3d.ymmin),
              nx=wp.w3d.nx, ny=wp.w3d.ny, nz=wp.w3d.nz, bz=bz)

########################
# Register Field Solvers
########################

# prevent GIST plotting from starting upon setup
wp.top.lprntpara = False
wp.top.lpsplots = False

if space_charge:
    # magnetostatic solver disabled for now, was causing mpi issues and is very slow
    # solverB = MagnetostaticMG()
    # solverB.mgtol = [0.01] * 3
    # registersolver(solverB)

    # Add 2D Field Solve, will be cylindrical solver based on setting of w3d.solvergeom
    solverE = wp.MultiGrid2D()
    wp.registersolver(solverE)

# Conductors must be registered after Field Solver is instantiated if we want them to impact field solve
for cond in conductors:
    wp.installconductor(cond)

# Conductors set as scrapers will remove impacting macroparticles from the simulation
#scraper = wp.ParticleScraper(conductors, lsavecondid=1)
scraper = wp.ParticleScraper(conductors)

######################
# Particle Diagnostics
######################

#wp.top.lsavelostparticles = True

# HDF5 Particle/Field diagnostic options

if particle_diagnostic_switch:
    particleperiod = 8000  # Particle diagnostic write frequency
    particle_diagnostic_0 = ParticleDiagnostic(period=particleperiod, top=wp.top, w3d=wp.w3d,  # Should always be set
                                               # Include data from all existing species in write
                                               species={species.name: species for species in wp.listofallspecies},
                                               # Option for parallel write (if available on system)
                                               comm_world=wp.comm_world, lparallel_output=False,
                                               # `ParticleDiagnostic` automatically append 'hdf5' to path name
                                               write_dir=diagDir[:-5])
    wp.installafterstep(particle_diagnostic_0.write)  # Write method is installed as an after-step action

if field_diagnostic_switch:
    fieldperiod = 8000  # Field diagnostic write frequency
    efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=wp.top, w3d=wp.w3d,
                                                              comm_world=wp.comm_world,
                                                              period=fieldperiod)
    wp.installafterstep(efield_diagnostic_0.write)

    # No B-Field diagnostic since magnetostatic solve turned off

    # bfield_diagnostic_0 = FieldDiagnostic.MagnetostaticFields(solver=solverB, top=top, w3d=w3d,
    #                                                           comm_world=comm_world,
    #                                                           period=fieldperiod)
    # installafterstep(bfield_diagnostic_0.write)

# Crossing Diagnostics
zcross_l = ZCrossingParticles(zz=z_positions[1], laccumulate=1)
zcross_r = ZCrossingParticles(zz=z_positions[4], laccumulate=1)


###########################
# Generate and Run PIC Code
###########################

electrons_tracked_t0 = wp.Species(type=wp.Electron)
tracer_count = 50

wp.derivqty()  # Set derived beam properties if any are required
wp.package("w3d")  # Use w3d solver/geometry package
wp.generate()  # Allocate arrays, generate mesh, perform initial field solve

loss_hist = []

#wp.restart('magnetized_cooler3200000')

Nsteps = 5000000
#particle_diagnostic_switch = False
#field_diagnostic_switch = False

while wp.top.it < Nsteps:

    wp.step(1000)

#        try:
#            np.save("trajectories_{}.npy".format(wp.top.it), electron_tracker_0.getsavedata())
#            electron_tracker_0.reset(clearhistory=1)
#        except:
#            pass
#        v_coords = np.ones([tracer_count, 3]) * beam_beta * wp.clight
#        v_coords[:, [0, 1]] = 0.0
#        x_vals = np.linspace(-beam_radius, beam_radius, tracer_count)
#        y_vals = np.zeros([tracer_count,])
#        z_vals = np.zeros(tracer_count) + 1e-3 
#        eptclArray = np.asarray([x_vals, v_coords[:,0], y_vals, v_coords[:,1], z_vals, v_coords[:,2]]).T
#        electron_tracker_0 = TraceParticle(js=electrons_tracked_t0.jslist[0],
#                                           x=eptclArray[:,0],
#                                           y=eptclArray[:,2],
#                                           z=eptclArray[:,4],
#                                           vx=np.zeros_like(eptclArray[:,0]),
#                                           vy=np.zeros_like(eptclArray[:,0]),
#                                           vz=eptclArray[:,5])

#    print("IONS H2+: {}".format(h2plus.getx().shape))
#    print("IONS e-: {}".format(emittedelec.getx().shape))
#    print("BEAM e-: {}".format(beam.getx().shape))
#    print("TRACER e-: {}".format(electron_tracker_0.getx().shape))

#    wp.step(100)

    vz_l_e = zcross_l.getvz(js=emittedelec.js)
    vz_l_h = zcross_l.getvz(js=h2plus.js)
    vz_r_e = zcross_r.getvz(js=emittedelec.js)
    vz_r_h = zcross_r.getvz(js=h2plus.js)

    loss_hist.append([wp.top.it, vz_l_e.size, vz_l_h.size, vz_r_e.size, vz_r_h.size])

#    if vz_l_e.size != 0 or vz_l_h.size != 0 or vz_r_e.size != 0 or vz_r_h.size != 0:
#        if wp.comm_world.rank == 0:
#            h5file =  h5.File(os.path.join('diags', 'crossing_record.h5'), 'a')
#            try:
#                l_group = h5file.create_group('left')
#                r_group = h5file.create_group('right')
#                l_e_group = l_group.create_group('e')
#                l_h_group = l_group.create_group('h')
#                r_e_group = r_group.create_group('e')
#                r_h_group = r_group.create_group('h')
#                l_e_group.attrs['position'] = zcross_l.zz
#                l_h_group.attrs['position'] = zcross_l.zz
#                r_e_group.attrs['position'] = zcross_r.zz
#                r_h_group.attrs['position'] = zcross_r.zz
#            except ValueError:
#                l_e_group = h5file['left/e']
#                l_h_group = h5file['left/h']
#                r_e_group = h5file['right/e']
#                r_h_group = h5file['right/h']

#            l_e_group.create_dataset('{}'.format(wp.top.it), data=vz_l_e)
#            l_h_group.create_dataset('{}'.format(wp.top.it), data=vz_l_h)
#            r_e_group.create_dataset('{}'.format(wp.top.it), data=vz_r_e)
#            r_h_group.create_dataset('{}'.format(wp.top.it), data=vz_r_h)

#            h5file.close()

    zcross_l.clear()
    zcross_r.clear()

    if wp.top.it % 100000 == 0:
        wp.dump()

if wp.comm_world.rank == 0 and wp.top.it == Nsteps:
    #print 'iteration = ', wp.top.it
    sample_times, curr_hist_i_r = \
    conductors[-1].get_current_history(
    js=h2plus.js,l_lost=1,l_emit=0,l_image=0,tmin=None,tmax=None,nt=100)
    sample_times, curr_hist_e_r = \
    conductors[-1].get_current_history(
    js=emittedelec.js,l_lost=1,l_emit=0,l_image=0,tmin=None,tmax=None,nt=100)
    sample_times, curr_hist_i_l = \
    conductors[0].get_current_history(
    js=h2plus.js,l_lost=1,l_emit=0,l_image=0,tmin=None,tmax=None,nt=100)
    sample_times, curr_hist_e_l = \
    conductors[0].get_current_history(
    js=emittedelec.js,l_lost=1,l_emit=0,l_image=0,tmin=None,tmax=None,nt=100)
    with open('curr_hist.txt', 'w') as fch:
        n = len(curr_hist_e_l)
        fch.write('{}\n'.format(n))
        for i in range(n):
            fch.write('{0} {1} {2} {3} {4}\n'.format(sample_times[i], curr_hist_e_l[i], curr_hist_i_l[i], curr_hist_e_r[i], curr_hist_i_r[i]))
    with open('loss_hist.txt', 'w') as flh:
        n = len(loss_hist)
        flh.write('{}\n'.format(n))
        for i in range(n):
            flh.write('{0} {1} {2} {3} {4}\n'.format(loss_hist[i][0], loss_hist[i][1], loss_hist[i][2], loss_hist[i][3], loss_hist[i][4]))

#if wp.comm_world.rank == 0:
#    print 'final global iteration = ', wp.top.it
