# Ionization of a Nitrogen gas sheet from a relativistic electron bunch 

# Beam parameters are based off of a proposed two-beam configuration at FACET II
# Drive bunch: Q = 1.4 nC, rms z = 20 um, sig-x = 50 um, sig-y = 27 um
# Witness bunch: Q = 0.5 nC, rms z = 14 um, sig-x = 5 um, sig-y = 7.5 um

#Ionization cross-sections are estimated from NIST tables - simplified model with minimal velocity spread
#Secondary ionization cross-section is estimated to be 1e2 larger but fixed

#This version separates out the initial ion species from the secondary ion species
#This assumption works so long as the primary ionization is "small" compared to the background gas population
#Ion macroparticle weights ae set to 1 for easy transfer.

# Nathan Cook
# 11/11/2021


# Basic imports
import sys, os, datetime, json
del sys.argv[1:] #  Necessry to run 'from warp import *' in IPython notebook without conflict.

# Import warp-specific packages
from warp import *
from warp.init_tools import *
from openpmd_viewer import OpenPMDTimeSeries
from warp.particles import tunnel_ionization

#ionization imports - only required if trying to use velocity dependent cross-sections for secondaries
#from ionization import Ionization
#from crosssections import H2IonizationEvent as h2xs

# Import rswarp packages -specify path if using custom install
sys.path.insert(2, '/home/vagrant/jupyter/rswarp/')
#import rswarp
from rswarp.utilities.file_utils import cleanupPrevious
from rswarp.utilities.file_utils import readparticles
from rswarp.utilities.file_utils import loadparticlefiles
from rswarp.ionization import ionization

# Import plotting and analysis packages
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Constants imports
from scipy.constants import e, m_e, c, k
from scipy.constants import c as clight
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e

# Set four-character run id, comment lines, user's name.
top.pline1    = "2D Witness"
top.runmaker  = "N Cook"
# Invoke setup routine for the plotting
setup()

# Set up the diagnostics and clean out the old files
diagDir = 'diags/hdf5/'
diagFDir = {'magnetic':'diags/fields','electric':'diags/fields'}

if comm_world.rank == 0:
    cleanupPrevious(diagDir,diagDir)

today = datetime.date.today().isoformat()
    

###################
###USER PARAMETERS
###################
#Domain parameters
L_X = 1.5e-3 #2.5e-3 #m
L_Y = 1.5e-3 #2.5e-3 #m
L_Z = 2.0e-3 #2.5e-3 #m

# Grid parameters
NUM_X = 196 #128 #256
NUM_Y = 196 #128 #256
NUM_Z = 256 #512

#Electrode and gas sheet parameters
EY_ELECTRODE = 1e4 #10 kV/m electric field between the electrodes. Used to define plate spacing
SHEET_DENSITY = 1e23 #number density in m^-3 - equivlent to 1e17 cm^-3
SHEET_WIDTH = 150e-6 #Width of the sheet along the z-axis
SHEET_OFFSET = 0.2e-3  
GAS_SPECIES = 'N2' #Species - either H2 or N2 - NOTE THIS DOESN'T HAVE A REAL EFFECT RIGHT NOW

#Set cross sections for ionization
PRIMARY_CROSS_SECTION = 2.65e-22 #Ionization cross section in m^-2 
SECONDARY_CROSS_SECTION = 2.65e-20 #Ionization cross section in m^-2

# Drive bunch: Q = 1.4 nC, rms z = 20 um, sig-x = 50 um, sig-y = 27 um
#Beam parameters
BEAM_CHARGE = 1.4e-9 #Beam charge in Coulombs
BEAM_ENERGY = 10. #10 GeV
BEAM_LENGTH = 20e-6 #100 micron beam ~ 0.3 ps
BEAM_DPP = 0.01 #1% dpp
BEAM_SIGMA_X = 50e-6 #100 micron transverse beam sigma
BEAM_SIGMA_Y = 27e-6
BEAM_EMIT = 20e-6 #20 um-rad normalized emittance

#Statistical/convergence parameters - these can be varied depending upon the gas sheet density
NUM_P = 5000000 #macroparticles in bunch
PRIMARY_WEIGHT = 1. #fix primary weight
SECONDARY_WEIGHT = PRIMARY_WEIGHT #1e2 #fix secondary weight

#Simulation parameters
N_STEPS = 200 #200 steps for simulation
FIELD_PERIOD = 200
PARTICLE_PERIOD = 200
FILE_PATH = 'diags/'

params = {}
params['SHEET_DENSITY'] = SHEET_DENSITY
params['SHEET_WIDTH'] = SHEET_WIDTH
params['GAS_SPECIES'] = GAS_SPECIES
params['BEAM_CHARGE'] = BEAM_CHARGE
params['BEAM_ENERGY'] = BEAM_ENERGY
params['BEAM_LENGTH'] = BEAM_LENGTH
params['BEAM_DPP'] = BEAM_DPP
params['BEAM_SIGMA_X'] = BEAM_SIGMA_X
params['BEAM_SIGMA_Y'] = BEAM_SIGMA_Y
params['BEAM_EMIT'] = BEAM_EMIT
params['NUM_P'] = NUM_P
params['PRIMARY_WEIGHT'] = PRIMARY_WEIGHT
params['SECONDARY_WEIGHT'] = SECONDARY_WEIGHT
params['PRIMARY_CROSS_SECTION'] = PRIMARY_CROSS_SECTION
params['SECONDARY_CROSS_SECTION'] = SECONDARY_CROSS_SECTION
params['N_STEPS'] = N_STEPS
params['L_X'] = L_X
params['L_Y'] = L_Y
params['L_Z'] = L_Z
params['NUM_X'] = NUM_X
params['NUM_Y'] = NUM_Y
params['NUM_Z'] = NUM_Z
params['DATE'] = today 
params['PATH'] = os.getcwd()

#dump params for quick search/reference
with open('params.txt', 'w') as outfile:
    json.dump(params, outfile, sort_keys=True,indent=4)
    
############################
# Domain / Geometry / Mesh #
############################

# Dimensions
X_MAX = L_X #L_X / 2.
X_MIN = 0.0 #-X_MAX
Y_MAX = L_Y #L_Y / 2.
Y_MIN = 0.0 #-Y_MAX
Z_MAX = L_Z #L_Z / 2.
Z_MIN = 0.0 #-Z_MAX

# cell sizes
dx = (X_MAX - X_MIN)/NUM_X
dy = (Y_MAX - Y_MIN)/NUM_Y
dz = (Z_MAX - Z_MIN)/NUM_Z

print("Domain specifications")
print(" --- (xmin, ymin, zmin) = ({}, {}, {})".format(X_MIN, Y_MIN, Z_MIN))
print(" --- (xmax, ymax, zmax) = ({}, {}, {})".format(X_MAX, Y_MAX, Z_MAX))
print(" --- (dx, dy, dz) = ({}, {}, {})".format(dx, dy, dz))


# Solver Geometry and Boundaries

# Specify solver geometry
w3d.solvergeom = w3d.XYZgeom

# Field boundary conditions

#differentiate between field being applied in z versus y
z_field = False

if z_field:
    #w3d.bounds specifies boundaries in order of lower, upper for x,y,z
    #0 = constant potential (dirichlet)
    #1 = zero normal derivative (neumann)
    #2 = periodic    
    w3d.bound0  = neumann
    w3d.boundnz = dirichlet
    w3d.boundxy = periodic
    
    #alternative approach is to specify each bound
    #w3d.bounds[0] = 1 #lower x
    #w3d.bounds[1] = 1 #upper x
    #w3d.bounds[2] = 1 #lower y
    #w3d.bounds[3] = 1 #upper y
    #w3d.bounds[4] = 0 #lower z
    #w3d.bounds[5] = 0 #upper z
    
else:
    #customize for y-directed field
    w3d.bound0  = dirichlet
    w3d.boundnz = dirichlet
    w3d.boundxy = dirichlet
    
    #alternative approach is to specify each bound
    #w3d.bounds[0] = 0 #lower x
    #w3d.bounds[1] = 0 #upper x
    #w3d.bounds[2] = 0 #lower y
    #w3d.bounds[3] = 0 #upper y
    #w3d.bounds[4] = 0 #lower z
    #w3d.bounds[5] = 0 #upper z

# Particles boundary conditions - absorb
top.pbound0 = absorb
top.pboundnz = absorb
top.pboundxy = absorb

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

# Set mesh cell sizes
w3d.dx = dx
w3d.dy = dy
w3d.dz = dz

#Define center points
zcen = (w3d.zmmax - w3d.zmmin)/2.0
xcen = (w3d.xmmax - w3d.xmmin)/2.0
ycen = (w3d.ymmax - w3d.ymmin)/2.0


#Set time step
#Define courant condition
if w3d.solvergeom==w3d.XZgeom:
    #2D Courant condition
    dtc = 1./np.sqrt((1./(w3d.dx/c)**2) + (1./(w3d.dz/c)**2))
    dt_want = dtc*0.99
else:
    #3D Courant condition
    dtc = 1./np.sqrt((1./(w3d.dx/c)**2) + (1./(w3d.dy/c)**2) + (1./(w3d.dz/c)**2))
    dt_want = dtc*0.99

#Set the appropriate time step BEFORE initializing the solver
top.dt = dt_want
print("Chosen time step: {}".format(top.dt))


DIM = '3d'
CIRC_M = 0
L_RZ = 0
    
# Current smoothing parameters
# ----------------------------
# Turn current smoothing on or off (0:off; 1:on)
use_smooth = 1 
# Number of passes of smoother and compensator in each direction (x, y, z)
npass_smooth = array([[ 1 , 1 ], [ 0 , 0 ], [ 1 , 1 ]])
# Smoothing coefficients in each direction (x, y, z)
alpha_smooth = array([[ 0.5, 3.], [ 0.5, 3.], [0.5, 3./2]])
# Stride in each direction (x, y, z)
stride_smooth = array([[ 1 , 1 ], [ 1 , 1 ], [ 1 , 1 ]])
set_smoothing_parameters( use_smooth, DIM, npass_smooth,
                         alpha_smooth, stride_smooth )


# Numerical parameters
# ---------------------
# Field solver (0:Yee, 1:Karkkainen on EF,B, 3:Lehe)
stencil = 0
# Particle shape (1:linear, 2:quadratic, 3:cubic)
depos_order = 2
# Gathering mode (1:from cell centers, 4:from Yee mesh)
efetch = 1
# Particle pusher (0:Boris, 1:Vay)
particle_pusher = 1


################
# FIELD SOLVER #
################
#No magnetic field in this simulation, but keeping this code here in case
magnetic_field = False
if magnetic_field:
    bz = np.zeros([w3d.nx, w3d.ny, w3d.nz])
    bz[:, :, :] = 200e-3
    z_start = w3d.zmmin
    z_stop = w3d.zmmax
    top.ibpush = 2
    addnewbgrd(z_start, z_stop, xs=w3d.xmmin, dx=(w3d.xmmax - w3d.xmmin), ys=w3d.ymmin, dy=(w3d.ymmax - w3d.ymmin),
                nx=w3d.nx, ny=w3d.ny, nz=w3d.nz, bz=bz)

# Set up fieldsolver
#Initialize the solver object - adding type_rz_depose=1
EM = EM3D(stencil=stencil,
    l_2drz=L_RZ,
    l_correct_num_Cherenkov = True,
    npass_smooth=npass_smooth,
    alpha_smooth=alpha_smooth,
    stride_smooth=stride_smooth)
    #l_setcowancoefs=1,
    #type_rz_depose=1)
registersolver(EM)

top.depos_order[...] = depos_order
top.pgroup.lebcancel_pusher = particle_pusher


###############################
# ELECTRON BEAM SETTINGS #
###############################
# Specify the maximum density of the e- drive bunch
n_electrons = BEAM_CHARGE / e  # number of electrons
sw_beam = n_electrons/NUM_P    # macroparticle weight
print("Number electrons = {}".format(n_electrons))

#Define RMS amplitudes for beam
w_x = BEAM_SIGMA_X
w_y = BEAM_SIGMA_Y
w_z = BEAM_LENGTH

#Define centroid parameters for beam
c_x = xcen
c_y = ycen
c_z = w3d.zmmin + 4.*BEAM_LENGTH #offset beam from minimum z by 4 sigma

#Compute gamma/beta, assuming BEAM_ENERGY is given in GeV
beam_gamma = BEAM_ENERGY*1e3/(0.510998910)
beta_gamma = sqrt(beam_gamma**2 - 1.)
beta_beam  = beta_gamma / beam_gamma

#Longitudinal momentum of the beam
beam_uz = beta_gamma
v_z_avg = beta_beam  * clight
u_z_avg = beta_gamma * clight

#Define RMS transverse velocity spread for beam
w_vx = (BEAM_EMIT/beta_gamma)/w_x
w_vy = (BEAM_EMIT/beta_gamma)/w_y

#Amplitude components
x_beam = np.random.normal(c_x, w_x, NUM_P)
y_beam = np.random.normal(c_y, w_y, NUM_P)
z_beam = np.random.normal(c_z, w_z, NUM_P)

#Velocity compoments
vx_beam = np.random.normal(0.0, w_vx, NUM_P)
vy_beam = np.random.normal(0.0, w_vy, NUM_P)
vz_beam = v_z_avg*np.ones(NUM_P) #Variations in velocity are irrelevant at high gamma


print("Beam macroparticle weight {} ".format(sw_beam))
print("Beam centered at [{},{},{}] mm".format(c_x*1e3,c_y*1e3,c_z*1e3))


beam = Species(type = Electron, name = 'beam')
beam.sw = sw_beam

L_RELATIVITY = True #Flag for relativistic corrections to kinematics
top.lrelativ = L_RELATIVITY

derivqty()

top.fstype=-1
package("w3d")
generate()

####################################
# Ionization of background gas     #
####################################


simulateIonization = True

if simulateIonization:
    #We will provide a density in 1/m^3, but one could precompute it via Temperature + Pressure
    #target_pressure = 5  # in Pa
    #target_temp = 273  # in K
    #target_density = target_pressure / boltzmann / target_temp  # in 1/m^3

    # These two species represent the primary (initial) emitted particles
    h2plus = Species(type=Dinitrogen, charge_state=+1, name='N2+', weight=PRIMARY_WEIGHT)
    elecprimary = Species(type=Electron, name='primary e-', weight=PRIMARY_WEIGHT)
    
    # These two species present the secondary emitted particles
    # Instantiate them with lower weight as there are usually fewer
    h2secondary = Species(type=Dinitrogen, charge_state=+1, name='N2second+', weight=SECONDARY_WEIGHT)
    elecsecondary = Species(type=Electron, name='secondary e-', weight=SECONDARY_WEIGHT)

    ioniz = ionization.Ionization(
        stride=100,
        xmin=w3d.xmmax * 0.25 ,
        xmax=w3d.xmmax * 0.75 ,
        ymin=w3d.ymmax * 0.25 ,
        ymax=w3d.ymmax * 0.75 ,
        zmin=w3d.zmmax * 0.25 ,
        zmax=w3d.zmmax * 0.75 ,
        nx=w3d.nx,
        ny=w3d.ny,
        nz=w3d.nz,
        l_verbose=True
    )

    #density = np.zeros([w3d.nx + 1, w3d.ny + 1, w3d.nz + 1])
    #dx, dy, dz = (w3d.xmmax / 2.  - w3d.xmmin / 2.) / w3d.nx, (w3d.ymmax / 2.  - w3d.ymmin / 2.) / w3d.ny, (w3d.zmmax / 2.  - w3d.zmmin / 2.) / w3d.nz
    #for i in range(w3d.nx):
    #    for j in range(w3d.ny):
    #        for k in range(w3d.nz):
    #            # # Define reservoir at a 45 degree angle
    #            xp, yp = i * dx - 0.002, j * dy - 0.002
    #            dist_x = (xp - yp) / 2.
    #            dist_y = (-xp + yp) / 2.
    #            if np.sqrt((xp - dist_x)**2 + (yp - dist_y)**2) < SHEET_WIDTH:
    #                density[i, j, k] = SHEET_DENSITY
                    
    #define arrays of the appropriate size
    custom_density = np.zeros([w3d.nx + 1, w3d.ny + 1, w3d.nz + 1])
    dens_zy = np.zeros([w3d.ny + 1, w3d.nz + 1])
    dx, dy, dz = w3d.dx, w3d.dy, w3d.dz

    #define width in terms of z-size
    nz_width = np.round(SHEET_WIDTH/(2.*dz))

    #fill in z-y profile
    for j in range(w3d.ny):
        for k in range(w3d.nz):
            if abs(j-k) < nz_width:
                dens_zy[j,k] = SHEET_DENSITY

    #copy to x positions
    for i in range(w3d.nx):
        custom_density[i,:,:] = dens_zy    
                
    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, elecprimary],
        cross_section=lambda vi: PRIMARY_CROSS_SECTION, #fixed crosssection in m^2
        emitted_energy0=[0, lambda vi, nnew: 1./np.sqrt(1.-((vi[0:nnew]/2.)/clight)**2) * emass*clight/jperev],
        emitted_energy_sigma=[0, 0],
        sampleEmittedAngle=lambda nnew, emitted_energy, incident_energy: np.random.uniform(0, 2*np.pi, size=nnew),
        writeAngleDataDir=False,
        writeAnglePeriod=1,
        l_remove_incident=False,
        l_remove_target=False,
        ndens = SHEET_DENSITY)
    
    ioniz.add(
        incident_species=elecprimary,
        emitted_species=[h2secondary, elecsecondary],
        cross_section=lambda vi: SECONDARY_CROSS_SECTION, #fixed crosssection in m^2
        emitted_energy0=[0, lambda vi, nnew: 1./np.sqrt(1.-((vi[0:nnew]/2.)/clight)**2) * emass*clight/jperev],
        emitted_energy_sigma=[0, 0],
        sampleEmittedAngle=lambda nnew, emitted_energy, incident_energy: np.random.uniform(0, 2*np.pi, size=nnew),
        writeAngleDataDir=False,
        writeAnglePeriod=1,
        l_remove_incident=False,
        l_remove_target=False,
        ndens = SHEET_DENSITY) #custom_density)
    
    ioniz.l_verbose = False #turn off verbosity



#####################
# Diagnostics Setup #
#####################
#Define and install field and particle diagnostics
#User defined FIELD_PERIOD and PARTICLE_PERIOD for each diagnostic

field_diag = FieldDiagnostic( period=FIELD_PERIOD, top=top, w3d=w3d, em=EM, comm_world=comm_world, lparallel_output=False)
installafterstep( field_diag.write )

particle_diagnostic_0 = ParticleDiagnostic(period=PARTICLE_PERIOD, top=top, w3d=w3d,
                                           species={species.name: species for species in listofallspecies},
                                           comm_world=comm_world, lparallel_output=False, write_dir=FILE_PATH)
installafterstep(particle_diagnostic_0.write)

####################
# CONTROL SEQUENCE #
####################

# prevent gist from starting upon setup
top.lprntpara = false
top.lpsplots = false

top.verbosity = 1      # Reduce solver verbosity

# add beam after calling generate
beam.addpart(x = x_beam, y = y_beam, z = z_beam, vx = vx_beam, vy = vy_beam, vz = vz_beam)

print(comm_world.rank)

for i in range(N_STEPS):
    step()
    
    #if (comm_world.rank == 0):
    h2y = h2plus.gety()
    h2sy = h2secondary.gety()
    print("Number of Steps: {}".format(i))
    print("Number of initial ions: {:.2e}".format(h2y.shape[0]*h2plus.sw))
    print("Number of secondary ions: {:.2e}".format(h2sy.shape[0]*h2secondary.sw))
    
#print timings
printtimers()

############
# ANALYSIS #
############

DO_ANALYSIS = True

if DO_ANALYSIS:
        
    print("Steps completed. Analysis beginning.")
    h2x = h2plus.getx()
    h2y = h2plus.gety()
    h2z = h2plus.getz()
    h2_ux = h2plus.getux()
    h2_uy = h2plus.getuy()
    h2_uz = h2plus.getuz()
    
    #secondaries
    hsx = h2plus.getx()
    hsy = h2plus.gety()
    hsz = h2plus.getz()
    hs_ux = h2plus.getux()
    hs_uy = h2plus.getuy()
    hs_uz = h2plus.getuz()
    
    print("Maximum y velocity: {}".format(np.max(h2_uy)))
    
    print("Number of primary macroparticles: {}".format(h2x.shape[0]))
    print("Number of primary ions: {}".format(h2x.shape[0]*h2plus.sw))
    
    print("Number of primary macroparticles: {}".format(h2x.shape[0]))
    print("Number of primary ions: {}".format(h2x.shape[0]*h2plus.sw))
    
    #center them
    gas_cz = np.mean(h2z)
    gas_cy = np.mean(h2y)
    gas_cx = np.mean(h2x)
    
    mod_H2p_x = h2x - gas_cx
    mod_H2p_y = h2y - gas_cy
    mod_H2p_z = h2z - gas_cz
    
    #subtract same centroid from secondaries
    mod_H2s_x = hsx - gas_cx
    mod_H2s_y = hsy - gas_cy
    mod_H2s_z = hsz - gas_cz
    
    #slice to only take those within gas sheet
    inds = np.where(np.abs(mod_H2p_z) < SHEET_WIDTH/2.)

    slice_H2p_x = mod_H2p_x[inds]
    slice_H2p_y = mod_H2p_y[inds]
    slice_H2p_z = mod_H2p_z[inds]
    
    slice_H2s_x = mod_H2s_x[inds]
    slice_H2s_y = mod_H2s_y[inds]
    slice_H2s_z = mod_H2s_z[inds]
    

    #u = beta*gamma
    #assume non-relativistic ions
    slice_H2p_vx = h2_ux[inds] * c
    slice_H2p_vy = h2_uy[inds] * c
    slice_H2p_vz = h2_uz[inds] * c
    
    slice_H2s_vx = hs_ux[inds] * c
    slice_H2s_vy = hs_uy[inds] * c
    slice_H2s_vz = hs_uz[inds] * c
    
    #combine arrays
    Hx_combined = np.hstack([slice_H2p_x,slice_H2s_x])
    Hy_combined = np.hstack([slice_H2p_y,slice_H2s_y])
    Hz_combined = np.hstack([slice_H2p_z,slice_H2s_z])
    Hvx_combined = np.hstack([slice_H2p_vx,slice_H2s_vx])
    Hvy_combined = np.hstack([slice_H2p_vy,slice_H2s_vy])
    Hvz_combined = np.hstack([slice_H2p_vz,slice_H2s_vz])
    
    
    num_macro = Hx_combined.shape[0]
    stride = 1
    if num_macro > 1e5:
        stride = 10
    
    #stack and transpose to produce coordinates as a seris of tuplets
    #N2coords = np.vstack([slice_H2p_x, slice_H2p_y, slice_H2p_z, slice_H2p_vx, slice_H2p_vy, slice_H2p_vz]).T
    N2coords = np.vstack([Hx_combined, Hy_combined, Hz_combined, Hvx_combined, Hvy_combined, Hvz_combined]).T
    header = " Impact Ionization with EM solver - singly ionized N2 molecules with weight {:.0e} from {:.0e} density sheet with stride {} \n Incident beam is {:.1f} nC bunch, sigmaz = {:.1f} micron, sigmax = {:.1f} micron, sigmay = {:.1f} micron \n Coordinates saved as: (x [m], y [m], z [m], vx [m/s], vy[m/s], vz [m/s]) \n {} ".format(SECONDARY_WEIGHT*stride,SHEET_DENSITY, stride, BEAM_CHARGE*1e9, BEAM_LENGTH*1e6, BEAM_SIGMA_X*1e6, BEAM_SIGMA_Y*1e6, today)
    np.savetxt("N2_EM_ions.txt", N2coords[::stride], header=header)
    
    
    #######################
    #Now plot configuration
    #######################
    fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(15,4))
    
    xlim = np.asarray([-0.125,0.125]) #np.asarray([X_MIN,X_MAX])*1e3-0.5
    zlim = np.asarray([-0.25,0.25]) #np.asarray([Z_MIN,Z_MAX])*1e3-1
    ylim = np.asarray([-0.125,0.125]) #np.asarray([Y_MIN,Y_MAX])*1e3-0.5
    
    #keep number of macro particles plotted in the thousands
    stride = 1
    if num_macro > 1e4:
        stride = 10
    elif num_macro > 1e5:
        stride = 100
    elif num_macro > 1e6:
        stride = 1000

    #ax1 - x vs. y
    ax1.scatter(slice_H2p_x[::stride]*1e3,slice_H2p_y[::stride]*1e3, s=4)
    ax1.set_xlabel(r'x [mm]', fontsize=12)
    ax1.set_ylabel(r'y [mm]', fontsize=12)
    ax1.set_title('H2+ primaries x-y',fontsize=15)
    ax1.set_xlim(xlim)
    ax1.set_ylim(ylim)
    
    #ax2 - z vs. x
    ax2.scatter(slice_H2p_z[::stride]*1e3,slice_H2p_x[::stride]*1e3, s=4)
    ax2.set_xlabel(r'z [mm]', fontsize=12)
    ax2.set_ylabel(r'x [mm]', fontsize=12)
    ax2.set_title('H2+ primaries z-x',fontsize=15)
    ax2.set_xlim(zlim)
    ax2.set_ylim(xlim)
    
    #ax3 - z vs. y
    ax3.scatter(slice_H2p_z[::stride]*1e3, slice_H2p_y[::stride]*1e3,s=4)
    ax3.set_xlabel(r'z [mm]', fontsize=12)
    ax3.set_ylabel(r'y [mm]', fontsize=12)
    ax3.set_title('H2+ primaries z-y',fontsize=15)
    ax3.set_xlim(zlim)
    ax3.set_ylim(ylim)
    
    fig.tight_layout()
    
    fig.savefig('sliced_primaries.png')