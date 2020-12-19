######################################################
#Test of STLconductor class with simple STL grid import
#Updated for Python 3 kernel on RadiaSoft container
#Yuan Hu and Nathan Cook
#12/18/2020
######################################################

import numpy as np
import h5py as h5
import time
import sys

# set warpoptions.ignoreUnknownArgs = True before main import to allow command line arguments
import warpoptions
warpoptions.ignoreUnknownArgs = True

import warp

# Users must edit this file
path_to_rswarp = '/home/vagrant/jupyter/rswarp/' #'/Users/yhu/Documents/Work/RadiaSoft/rswarp'
if not path_to_rswarp in sys.path: sys.path.insert(1, path_to_rswarp)

from copy import deepcopy
from random import randint
from rswarp.cathode import sources
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.file_utils import cleanupPrevious
from rswarp.diagnostics.parallel import save_lost_particles
from rswarp.diagnostics.ConductorDiagnostics import analyze_scraped_particles

from rswarp.stlconductor.stlconductor import STLconductor

# Constants imports
from scipy.constants import e, m_e, c, k

# Set matplotlib backend for saving plots (if requested)
import matplotlib as mpl 
  
import matplotlib.pyplot as plt 
from matplotlib import colors
from matplotlib import cm as cmaps

kb_eV = 8.6173324e-5  # Bolztmann constant in eV/K
kb_J = k  # Boltzmann constant in J/K
m = m_e  # electron mass



############################
# Domain / Geometry / Mesh #
############################

PLATE_SPACING = 5e-6     # plate spacing
CHANNEL_WIDTH = 6.900e-6  # width of simulation box

# Dimensions
X_MAX = +CHANNEL_WIDTH / 2
X_MIN = -X_MAX
Y_MAX = +CHANNEL_WIDTH / 2.
Y_MIN = -Y_MAX
Z_MAX = PLATE_SPACING
Z_MIN = 0.

# Grid parameters
NUM_X = 500; NUM_Y = 500; NUM_Z = 250

# z step size
dx = (X_MAX - X_MIN)/NUM_X
dy = (Y_MAX - Y_MIN)/NUM_Y
dz = (Z_MAX - Z_MIN)/NUM_Z

# Solver Geometry and Boundaries

# Specify solver geometry
warp.w3d.solvergeom = warp.w3d.XYZgeom

# Set field boundary conditions
warp.w3d.bound0 = warp.neumann
warp.w3d.boundnz = warp.dirichlet
warp.w3d.boundxy = warp.periodic

# Particles boundary conditions
warp.top.pbound0 = warp.absorb
warp.top.pboundnz = warp.absorb
warp.top.pboundxy = warp.periodic

# Set mesh boundaries
warp.w3d.xmmin = X_MIN
warp.w3d.xmmax = X_MAX
warp.w3d.ymmin = Y_MIN
warp.w3d.ymmax = Y_MAX
warp.w3d.zmmin = 0.
warp.w3d.zmmax = Z_MAX

# Set mesh cell counts
warp.w3d.nx = NUM_X
warp.w3d.ny = NUM_Y
warp.w3d.nz = NUM_Z

###############################
# PARTICLE INJECTION SETTINGS #
###############################

injection_type = 1
cathode_temperature = 1273.15
cathode_workfunction = 2.0             # in eV
anode_workfunction = 0.1
volts_on_conductor = 10.

# INJECTION SPECIFICATION
USER_INJECT = injection_type

# Cathode and anode settings
CATHODE_TEMP = cathode_temperature
CATHODE_PHI = cathode_workfunction
ANODE_WF = anode_workfunction          # Can be used if vacuum level is being set
CONDUCTOR_VOLTS = volts_on_conductor   # ACCEL_VOLTS used for velocity and CL calculations

# Emitted species
# Emitter area and position
SOURCE_RADIUS_1 = 0.5 * CHANNEL_WIDTH  # a0 parameter - X plane
SOURCE_RADIUS_2 = 0.5 * CHANNEL_WIDTH  # b0 parameter - Y plane
Z_PART_MIN = dz / 1000.                # starting particle z value

# Compute cathode area for geomtry-specific current calculations
if (warp.w3d.solvergeom == warp.w3d.XYZgeom):
    # For 3D cartesion geometry only
    cathode_area = 4. * SOURCE_RADIUS_1 * SOURCE_RADIUS_2
else:
    # Assume 2D XZ geometry
    cathode_area = 2. * SOURCE_RADIUS_1 * 1.

# If using the XZ geometry, set so injection uses the same geometry
warp.top.linj_rectangle = (warp.w3d.solvergeom == warp.w3d.XZgeom or warp.w3d.solvergeom == warp.w3d.XYZgeom)

PTCL_PER_STEP = 300
CURRENT_MODIFIER = 0.5                 # Factor to multiply CL current by when setting beam current

warp.derivqty()

################
# FIELD SOLVER #
################

# Set up fieldsolver
warp.f3d.mgtol = 1e-6
solverE = warp.MultiGrid3D()
warp.registersolver(solverE)

##########################
# CONDUCTOR INSTALLATION #
##########################

install_conductor = True
ofile_prefix = "simple_tec_grid__dx{:.2f}".format(dz*1e9)

raytri = 'watertight'#'moller' #Specify the ray-triangle intersection scheme. Choose between `moller` and `watertight`
ofile_prefix = "circular_apperture_array_dx{:.2f}_{}".format(dz*1e9,raytri)
install_conductor = True

if install_conductor:
    if raytri == 'watertight':
        #No further manipulation needed
        conductor = STLconductor("../grid/simple_tec_grid.stl", raytri_scheme=raytri, verbose="on", voltage=CONDUCTOR_VOLTS, normalization_factor=dz, condid=1)
    else:
        #Likely need to add a small displacement to prevent errors in the intercept calculation
        conductor = STLconductor("../grid/simple_tec_grid.stl", raytri_scheme=raytri, disp='auto', verbose="on", voltage=CONDUCTOR_VOLTS, normalization_factor=dz, condid=1)
        
# --- Anode Location
zplate = Z_MAX

# Create source conductors
if install_conductor:
    source = warp.ZPlane(zcent=warp.w3d.zmmin, zsign=-1., voltage=0., condid=2)
else:
    source = warp.ZPlane(zcent=warp.w3d.zmmin, zsign=-1., voltage=0.)

# Create ground plate
if install_conductor:
    plate = warp.ZPlane(voltage=0., zcent=zplate, condid=3)
else:
    plate = warp.ZPlane(voltage=volts_on_conductor, zcent=zplate)


if install_conductor :
    warp.installconductor(conductor, dfill=warp.largepos)
    warp.installconductor(source, dfill=warp.largepos)
    warp.installconductor(plate, dfill=warp.largepos)
    scraper = warp.ParticleScraper([source, plate],
                              lcollectlpdata=True,
                              lsaveintercept=True)
    scraper_dictionary = {1: 'source', 2: 'collector'}
else:
    warp.installconductor(source, dfill=warp.largepos)
    warp.installconductor(plate, dfill=warp.largepos)
    scraper = warp.ParticleScraper([source, plate])
    scraper_dictionary = {1: 'source', 2: 'collector'}
    
####################
# CONTROL SEQUENCE #
####################

# prevent gist from starting upon setup
warp.top.lprntpara = False
warp.top.lpsplots = False

warp.top.verbosity = 1      # Reduce solver verbosity
solverE.mgverbose = 1  # further reduce output upon stepping - prevents websocket timeouts in Jupyter notebook

init_iters = 20000
regular_iters = 200

init_tol = 1e-6
regular_tol = 1e-6

# Time Step

# initialize field solver and potential field
solverE.mgmaxiters = init_iters
solverE.mgtol = init_tol
warp.package("w3d")
warp.generate()

warp.step(1)


####################
# ANALYSIS & PLOTS #
####################

phi = solverE.getphi()
E = np.sqrt(solverE.getez() ** 2 + solverE.getex() ** 2 + solverE.getey() ** 2)
Ex = solverE.getex()

# xy-plane at z = zcent
grid_z = 3.5e-6
grid_iz = int((grid_z-Z_MIN)/dz)

# Phi on yz-plane at x = xcent
grid_x  = X_MIN + CHANNEL_WIDTH*0.5
grid_ix = int((grid_x-X_MIN)/dx)
grid_y  = Y_MIN + CHANNEL_WIDTH*0.5
grid_iy = int((grid_y-Y_MIN)/dy)

x = np.linspace(X_MIN,X_MAX,NUM_X+1)*1e6
y = np.linspace(Y_MIN,Y_MAX,NUM_Y+1)*1e6
z = np.linspace(Z_MIN,Z_MAX,NUM_Z+1)*1e6

X, Y = np.meshgrid(x,y)


#Plot E at grid (slice XY-plane)
fig,ax = plt.subplots()
cf1 = ax.contourf(X,Y,E[:,:,grid_iz]*1e-6,20,cmap = cmaps.viridis)
ax.set_xlabel(r'x position [$\mu$m]')
ax.set_ylabel(r'y position [$\mu$m]')
fig.colorbar(cf1, label = '[MV/m]')
ax.set_aspect(1)

fig.savefig(ofile_prefix+"_efield.png")



#Plot Phi across domain (slice ZY-plane)
Z, Y = np.meshgrid(z,y)

fig,ax = plt.subplots()
cf2 = ax.contourf(Z, Y, phi[grid_ix,:,:],20,cmap = cmaps.viridis)
ax.set_xlabel(r'x position [$\mu$m]')
ax.set_ylabel(r'y position [$\mu$m]')
clim = [0., 10.]
cf2.set_clim(clim[0], clim[1])
fig.colorbar(cf2,label = 'Volts', ticks = np.arange(clim[0], clim[1]+1e-10, 2))
ax.set_aspect('auto')

fig.savefig(ofile_prefix+"phi_transverse.png")