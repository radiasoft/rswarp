#Tracking ions through an ion microscope design, using fields generated from a CST file and reorganized to be loaded into Warp
#The simulation uses a moving window to transport ions centered at z = 0 to a monitor at z = 170 mm
#Include z-crossing diagnostic to capture transverse coordinates at 170 mm and write to file

# Basic imports
import sys, os, datetime, json
del sys.argv[1:] #  Necessry to run 'from warp import *' in IPython notebook without conflict.

# Import warp-specific packages
from warp import *
import warp as wp
from warp.init_tools import *
from openpmd_viewer import OpenPMDTimeSeries
from warp.particles import tunnel_ionization
from warp.particles.extpart import ZCrossingParticles


# Import rswarp packages -specify path if using custom install
sys.path.insert(2, '/home/vagrant/jupyter/rswarp/rswarp/')
import rswarp
from rswarp.utilities.file_utils import cleanupPrevious
from rswarp.utilities.file_utils import readparticles
from rswarp.utilities.file_utils import loadparticlefiles
from rswarp.diagnostics import FieldDiagnostic


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

# DOMAIN parameters
L_Z = 40.0e-3 #2.0e-3 #m
L_X = 20.0e-3 #0e-3 #m
L_Y = 20.0e-3 #m

# Grid parameters
NUM_X = int(L_X/5e-3)*32#64#32
NUM_Y = int(L_Y/5e-3)*32#64#32 
NUM_Z = int(L_Z/5e-3)*16#32#16

#particle weight adjustment
ADJUST_WEIGHTS = 0.1 #this effectively reduces gas density by a factor of 10
DO_DEPOSITION = 1

#sheet parameters
SHEET_WIDTH = 150e-6 #Width of the sheet along the z-axis
SHEET_ANGLE = np.pi/4 #angle w.r.t z/y axis
SHEET_Z0 = 1.25e-3 #centroid of gas jet along beam-axis

#PATHS FOR LOADING FIELDS AND IONS
ION_PATH = '/home/vagrant/jupyter/StaffScratch/ncook882/GSM/datasets/impact/drive_1e20/'
FIELD_PATH = '/home/vagrant/jupyter/StaffScratch/ncook882/GSM/externalfields/'

#diagnostic parameters
FIELD_PERIOD = 500
PARTICLE_PERIOD = 500
FILE_PATH = 'diags'

#Simulation parameters
N_STEPS = 6000 #2500 steps for simulation
Z_MONITOR = 170e-3 #location of monitor
V0_WINDOW = 1.5e4 #m/s in the z-direction (by default)
WINDOW_STEPS = 100
TIME_STEP = 5e-10
TIME_STEP_FACTOR = 0.4



params = {}
params['ION_PATH'] = ION_PATH
params['FIELD_PATH'] = FIELD_PATH
params['Z_MONITOR'] = Z_MONITOR
params['V0_WINDOW'] = V0_WINDOW
params['ADJUST_WEIGHTS'] = ADJUST_WEIGHTS
params['N_STEPS'] = N_STEPS
params['TIME_STEP'] = TIME_STEP
params['TIME_STEP_FACTOR'] = TIME_STEP_FACTOR
params['WINDOW_STEPS'] = WINDOW_STEPS
params['SHEET_WIDTH'] = SHEET_WIDTH
params['SHEET_ANGLE'] = SHEET_ANGLE
params['SHEET_Z0'] = SHEET_Z0
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
X_MAX = L_X/2 #L_X / 2.
X_MIN = -L_X/2. #0.0 #-X_MAX
Y_MAX = L_Y/2 #L_Z / 2.
Y_MIN = -L_Y/2. #0.0 #-Z_MAX
Z_MAX = L_Z/2. #95.*L_Z/100. #L_Y / 2.
Z_MIN = -L_Z/2. #-5.*L_Z/100. #-Y_MAX


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
#w3d.bounds specifies boundaries in order of lower, upper for x,y,z
#0 = constant potential (dirichlet)
#1 = zero normal derivative (neumann)
#2 = periodic

w3d.bound0  = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = dirichlet

print(w3d.bounds)

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


# Set time step
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
top.dt = TIME_STEP #1.2e-10 #dt_want*2000.
print("Chosen time step: {}".format(top.dt))

#Set the moving window
top.vbeamfrm = V0_WINDOW

#Set the crossing diagnostic
zcross_monitor = ZCrossingParticles(zz=Z_MONITOR, laccumulate=1)

################
# FIELD SOLVER #
################

# Set up fieldsolver
f3d.mgtol = 1e-6
solverE = MultiGrid3D()
registersolver(solverE)

# prevent gist from starting upon setup
top.lprntpara = false
top.lpsplots = false

top.verbosity = 1      # Reduce solver verbosity
solverE.mgverbose = 1  # further reduce output upon stepping - prevents websocket timeouts in Jupyter notebook

init_iters = 2000
regular_iters = 100

init_tol = 1e-6
regular_tol = 1e-6

# initialize field solver and potential field
solverE.mgmaxiters = init_iters
solverE.mgtol = init_tol


######################
# DEFINE ION SPECIES #
######################
ions = Species(type = Dinitrogen, name = 'N2+', charge_state=+1) #set to have +1 charge
ions.sw = ADJUST_WEIGHTS #stand-in for weights

L_RELATIVITY = True #Flag for relativistic corrections to kinematics
top.lrelativ = L_RELATIVITY

#set vbeam
ions.vbeam = V0_WINDOW

derivqty()

#####################
# Diagnostics Setup #
#####################
#Define and install field and particle diagnostics
#User defined FIELD_PERIOD and PARTICLE_PERIOD for each diagnostic
#efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d,
#                                            comm_world=comm_world,period=FIELD_PERIOD, 
#                                            write_dir=os.path.join(FILE_PATH,'fields'))
#installafterstep(efield_diagnostic_0.write)

particle_diagnostic_0 = ParticleDiagnostic(period=PARTICLE_PERIOD, top=top, w3d=w3d,
                                           species={species.name: species for species in listofallspecies},
                                           comm_world=comm_world, lparallel_output=False, write_dir=FILE_PATH)
installafterstep(particle_diagnostic_0.write)


################
# EXTERNAL FIELD #
################

#load external field
ExWarp = np.load(FIELD_PATH+'rotEx15.npy')
EyWarp = np.load(FIELD_PATH+'rotEy15.npy')
EzWarp = np.load(FIELD_PATH+'rotEz15.npy')

nx_E, ny_E, nz_E = ExWarp.shape

#load coordinates for field
x_range = np.load(FIELD_PATH+'rotx15.npy')
y_range = np.load(FIELD_PATH+'roty15.npy')
z_range = np.load(FIELD_PATH+'rotz15.npy')

x_min = x_range[0]
y_min = y_range[0]
z_min = z_range[0]

dx_E = x_range[1]-x_range[0]
dy_E = dx_E #0
dz_E = z_range[1]-z_range[0]


print("Field specified")

#now add new grid      
wp.addnewegrd(z_range[0], z_range[-1],
              xs = x_range[0], dx = dx_E,
              ys = y_range[0], dy = dy_E,
              nx = nx_E, ny = ny_E, nz = nz_E,
              ex = ExWarp, ey = EyWarp, ez = EzWarp)

print("Added new egrid")


#############
# LOAD IONS #
#############

#Load ion data from hdf5 file

analysis_folder = os.path.join(ION_PATH,'diags/hdf5/')
ts = OpenPMDTimeSeries(analysis_folder)
num_files = ts.iterations.shape[0]
index = num_files-1
ind_num = ts.iterations[index]
iter_time = ts.t[index]

#reconcile differences in species names
if 'N2+' in ts.avail_species:
    prim_spec = 'N2+'
    sec_spec = 'N2second+'
else:
    prim_spec = 'H2+'
    sec_spec = 'H2second+'

#get initial ion particles
prim_z, prim_x, prim_y, prim_ux, prim_uy, prim_uz, prim_w = ts.get_particle(var_list=['z','x','y','ux','uy','uz','w'],species=prim_spec, iteration=ind_num)

#get secondary ion particles
sec_z, sec_x, sec_y, sec_ux, sec_uy, sec_uz, sec_w = ts.get_particle(var_list=['z','x','y','ux','uy','uz','w'],species=sec_spec, iteration=ind_num)


#reset centroids and slice along z
#take primaries within SHEET_WIDTH of centroid
#secondaries can be within 1.25*SHEET_WIDTH
prim_inds = np.where(np.abs(prim_z - SHEET_Z0)<SHEET_WIDTH)
sec_inds = np.where(np.abs(sec_z - SHEET_Z0)<1.25*SHEET_WIDTH/2.)

#combine primaries and secondaries
z_ions = np.hstack([prim_z[prim_inds],sec_z[sec_inds]])-SHEET_Z0
x_ions = np.hstack([prim_x[prim_inds],sec_x[sec_inds]])
y_ions = np.hstack([prim_y[prim_inds],sec_y[sec_inds]])

#reset x centroid as well
ions_cx = np.mean(x_ions)
x_ions = x_ions - ions_cx

## Now preview what it will look like when z and y are flipped for tracking
#rotate in z-y plane by constructing an offset and adding it to the z values
y_offsets = y_ions-np.mean(y_ions) #compute vertical offset
z_offsets = -1.*y_offsets #positive y offset should correlate to negative z offset for coordinate transform
z_ions = z_ions +z_offsets #add offset

vz_ions = np.hstack([prim_uz[prim_inds],sec_uz[sec_inds]])/(ions.mass*c)
vx_ions = np.hstack([prim_ux[prim_inds],sec_ux[sec_inds]])/(ions.mass*c)
vy_ions = np.hstack([prim_uy[prim_inds],sec_uy[sec_inds]])/(ions.mass*c)
sw_ions = np.hstack([prim_w[prim_inds],sec_w[sec_inds]])*ADJUST_WEIGHTS #adjust weights

if comm_world.rank == 0:
    #now preview the microscope frame particles
    with mpl.style.context('rs_paper'):

        fig,(ax1,ax2,ax3) = plt.subplots(1,3, figsize=(24,6))

        #ax1 - z (y) vs. x
        ax1.scatter(y_ions*1e3,x_ions*1e3, s=4)
        ax1.set_xlabel(r'z [mm]', fontsize=12)
        ax1.set_ylabel(r'x [mm]', fontsize=12)
        ax1.set_title('z-x angled profile - microscope sim frame',fontsize=15)
        ax1.set_xlim(-1,1)
        ax1.set_ylim(-1,1)

        #ax2 - x vs. y
        ax2.scatter(x_ions*1e3,z_ions*1e3, s=4)
        ax2.set_xlabel(r'x [mm]', fontsize=12)
        ax2.set_ylabel(r'y [mm]', fontsize=12)
        ax2.set_title('x-y angled profile - microscope sim frame',fontsize=15)
        ax2.set_xlim(-1,1)
        ax2.set_ylim(-1,1)

        #ax2 - z vs. y
        ax3.scatter(y_ions*1e3,z_ions*1e3, s=4)
        ax3.set_xlabel(r'z [mm]', fontsize=12)
        ax3.set_ylabel(r'y [mm]', fontsize=12)
        ax3.set_title('z-y angled profile - microscope sim frame',fontsize=15)
        ax3.set_xlim(-1,1)
        ax3.set_ylim(-1,1)

        fig.savefig('transport_transformed_ions.png')


package("w3d")
generate()

# add beam after calling generate
#NOTE THAT WE NEED TO SWITCH THE Y AND Z COORDINATES HERE!!!!
#THE FIELDS ARE DEFINED ALONG Z, WHICH WAS PREVIOUSLY THE BEAM AXIS!!!!
ions.addparticles(x = x_ions, y = z_ions, z = y_ions, vx = vx_ions , vy = vz_ions, vz = vy_ions)

#Set the moving window (might need to be done now)
top.vbeamfrm = V0_WINDOW

#Set deposition to 0 to cancel space charge -> not sure this is working properly
top.pgroup.ldodepos[0] = DO_DEPOSITION

print(top.vbeamfrm)

####################
# CONTROL SEQUENCE #
####################

new_zmin = w3d.zmmin
new_zmax = w3d.zmmax

elapsed_time = 0.

ion_count = x_ions.shape[0]

#array to store transmission and times
times = []
zvals = []
trans = []
xsigs = []
ysigs = []

#loop through N_STEPS but also prevent beam being accelerated back towards beampipe
while (top.it < N_STEPS) & (top.vbeamfrm > 0) :
    #step a small amount
    step(WINDOW_STEPS)
    
    #update effective window
    new_zmin = new_zmin + top.vbeamfrm*top.dt*WINDOW_STEPS
    new_zmax = new_zmax + top.vbeamfrm*top.dt*WINDOW_STEPS
    
    #update elapsed time
    elapsed_time = elapsed_time + top.dt*WINDOW_STEPS
    
    #compute new window speed from z velocity average
    vz = ions.getvz()
    mean_vz = np.mean(vz)
    peak_vz = np.max(vz)
    top.vbeamfrm = mean_vz
    
    #grab mean z
    z = ions.getz()
    mean_z = np.mean(z)
    
    x = ions.getx()
    y = ions.gety()
    
    xsigs.append(np.std(x))
    ysigs.append(np.std(y)) 
    
    #compute new timestep
    dt_max = w3d.dy/peak_vz
    top.dt = dt_max*0.4 #set new timestep
    
    print("Completed step: {}".format(top.it))
    print("New window: {} - {} mm".format(new_zmin*1e3,new_zmax*1e3))
    print("Setting new window velocity to: {} m/s".format(top.vbeamfrm))
    print("Setting new timestep to: {} ns".format(top.dt*1e9))
    
    
    n_ions = vz.shape[0]
    times.append(elapsed_time)
    trans.append(n_ions/ion_count)
    zvals.append(mean_z)
    

#print timings
printtimers()

z = ions.getz()
y = ions.gety()
x = ions.getx()

try:
    vz = ions.getvz()
    print("Max vz: {}".format(np.max(vz)))
except:
    print("No vz info")

print("x extent: {} - {}".format(np.min(x),np.max(x)))
print("y extent: {} - {}".format(np.min(y),np.max(y)))
print("z min: {}".format(np.min(z)))
print("z max: {}".format(np.max(z)))

final_ion_count = x.shape[0]
transmission = final_ion_count/ion_count
print("Transmission: {}".format(transmission))

#print(ions.getweights())
#top.it is the current step/iteration #

try:
    #get the z crossing data
    xcross = zcross_monitor.getx()
    ycross = zcross_monitor.gety()
    tcross = zcross_monitor.gett()
    
    print("Mean crossing time: {}".format(np.mean(tcross)))

    if comm_world.rank == 0:
        #save the x/y particle distribution at the zcrossing
        np.save('xy.npy',np.vstack([xcross,ycross]))

        with mpl.style.context('rs_paper'):
        
            fig,ax1 = plt.subplots(1,1, figsize=(8,6))
        
            #ax1 - x vs. y
            ax1.scatter(x*1e3,y*1e3, s=4)
            ax1.set_xlabel(r'x [mm]', fontsize=12)
            ax1.set_ylabel(r'y [mm]', fontsize=12)
            ax1.set_title('x-y profile at z={} crossing'.format(Z_MONITOR),fontsize=15)
            ax1.set_xlim(w3d.xmmin*1e3,w3d.xmmax*1e3)
            ax1.set_ylim(w3d.ymmin*1e3,w3d.ymmax*1e3)

            fig.savefig('crossing_{}.png'.format(N_STEPS))

except:
    pass
    
    
if comm_world.rank == 0:
    
    np.save('xysigs.npy',np.vstack([zvals,np.asarray(xsigs),np.asarray(ysigs)]))
    
    with mpl.style.context('rs_paper'):
        
        fig,ax1 = plt.subplots(1,1, figsize=(8,6))
        
        #ax1 - transmission vs. time
        ax1.plot(np.asarray(times)*1e9,np.asarray(trans)*100)
        ax1.set_xlabel(r't [ns]', fontsize=12)
        ax1.set_ylabel(r'transmission [%]', fontsize=12)
        ax1.set_xlim(0,elapsed_time*1e9)
        ax1.set_ylim(0,105)
        
        fig.savefig('time_transmission_{}.png'.format(N_STEPS))
        
    with mpl.style.context('rs_paper'):
        
        fig,ax1 = plt.subplots(1,1, figsize=(8,6))
        
        #ax1 - transmission vs. z
        ax1.plot(np.asarray(zvals)*1e3,np.asarray(trans)*100)
        ax1.set_xlabel(r'z [mm]', fontsize=12)
        ax1.set_ylabel(r'transmission [%]', fontsize=12)
        ax1.set_xlim(0,Z_MONITOR*1e3)
        ax1.set_ylim(0,105)
        
        fig.savefig('z_transmission_{}.png'.format(N_STEPS))
        
    with mpl.style.context('rs_paper'):
        
        fig,(ax1,ax2) = plt.subplots(1,2, figsize=(16,6))
        
        #ax1 - x vs. z
        ax1.scatter(x*1e3,z*1e3, s=4)
        ax1.set_xlabel(r'x [mm]', fontsize=12)
        ax1.set_ylabel(r'z [mm]', fontsize=12)
        ax1.set_title('x-z profile at t = {:.2f} ns'.format(elapsed_time*1e9),fontsize=15)
        ax1.set_xlim(w3d.xmmin*1e3,w3d.xmmax*1e3)
        ax1.set_ylim(new_zmin*1e3,new_zmax*1e3)
        
        #ax2 - x vs. y
        ax2.scatter(x*1e3,y*1e3, s=4)
        ax2.set_xlabel(r'x [mm]', fontsize=12)
        ax2.set_ylabel(r'y [mm]', fontsize=12)
        ax2.set_title('x-y profile at t = {:.2f} ns'.format(elapsed_time*1e9),fontsize=15)
        ax2.set_xlim(w3d.xmmin*1e3,w3d.xmmax*1e3)
        ax2.set_ylim(w3d.ymmin*1e3,w3d.ymmax*1e3)
        
        fig.savefig('ions_{}dt-variabletimestep-lowercharge.png'.format(N_STEPS))