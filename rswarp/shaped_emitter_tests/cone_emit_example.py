# # Test script simulating a 3D dielectric sphere between two parallel plates. This script should only be run on a single core.
# 
# User Flags:
# l_MR - If set, then runs with mesh refinement
# USE_3D - If set, then runs with a 3D geometry rather than 2D.
# MAKE_PLOTS - If set, plots a 2D slice of the potential and compares a 1D lineout to the analytic solution.
# USE_GIST - If set, then creates gist plots as well
#
# 01/09/2018
#  
# Jean-Luc Vay and Nathan Cook

from __future__ import division
import sys
import os
import time

import matplotlib as mpl
mpl.use('Agg')
#mpl.rcParams.update({'font.size': 16})

del sys.argv[1:]

from warp import * 
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.data_dumping.openpmd_diag import ElectrostaticFieldDiagnostic
from warp.particles.extpart import ZCrossingParticles

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

from scipy.constants import epsilon_0 as e0
from scipy.constants import m_e as me 
from scipy.constants import elementary_charge as q
from scipy.constants import k as kb

import emitters

# --- user flags
l_MR       = True #IF true, use mesh refinement
USE_3D     = True #If true, use 3D solver
MAKE_PLOTS = True #If true, plots a 2D slice of the potential and compares a 1D lineout to the analytic solution

# Useful utility function - See the rswarp repository for further details:
# https://github.com/radiasoft/rswarp/blob/master/rswarp/utilities/file_utils.py
def cleanupPrevious(particleDirectory, fieldDirectory):

    """
    Remove old diagnostic files.

    Parameters:
            particleDirectory (str): Path to particle diagnostics

    """
    if os.path.exists(particleDirectory):
        files = os.listdir(particleDirectory)
        for file in files:
            if file.endswith('.h5'):
                os.remove(os.path.join(particleDirectory,file))
    if isinstance(fieldDirectory,dict):
        for key in fieldDirectory:
            if os.path.exists(fieldDirectory[key]):
                files = os.listdir(fieldDirectory[key])
                for file in files:
                    if file.endswith('.h5'):
                        os.remove(os.path.join(fieldDirectory[key],file))
    elif isinstance(fieldDirectory, list):
        for directory in fieldDirectory:
            if os.path.exists(directory):
                files = os.listdir(directory)
                for file in files:
                    if file.endswith('.h5'):
                        os.remove(os.path.join(directory, file))
    elif isinstance(fieldDirectory, str):
            if os.path.exists(fieldDirectory):
                files = os.listdir(fieldDirectory)
                for file in files:
                    if file.endswith('.h5'):
                        os.remove(os.path.join(fieldDirectory, file))


    

# Constants imports
from scipy.constants import e, m_e, c, k
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e

diagDir = 'diags7/xzsolver/hdf5/'
field_base_path = 'diags7/fields/'
diagFDir = {'magnetic':'diags7/fields/magnetic','electric':'diags7/fields/electric'}

# Cleanup previous files
if comm_world.rank == 0:
    cleanupPrevious(diagDir,diagFDir)

if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=False, error=False)

#print "rank:", comm_world.rank

#Dimensions

PLATE_SPACING = 1.e-6 #plate spacing
CHANNEL_WIDTH = 1.e-6 #width of simulation box

X_MAX = CHANNEL_WIDTH*0.5
X_MIN = -1.*X_MAX
Y_MAX = CHANNEL_WIDTH*0.5
Y_MIN = -1.*Y_MAX
Z_MIN = 0.
Z_MAX = PLATE_SPACING


#Grid parameters - increase number of cells if running in parallel

N_ALL = 32
NUM_X = N_ALL
NUM_Y = N_ALL
NUM_Z = N_ALL


## Solver Geometry

# Set boundary conditions
w3d.bound0  = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = periodic
top.pbound0 = absorb
top.pboundnz = absorb

top.wpid = nextpid()
top.ssnpid = nextpid()
top.xbirthpid = nextpid()
top.ybirthpid = nextpid()

# Set mesh boundaries
w3d.xmmin = X_MIN
w3d.xmmax = X_MAX
w3d.zmmin = 0.
w3d.zmmax = Z_MAX

# Set mesh cell counts
w3d.nx = NUM_X
w3d.nz = NUM_Z

w3d.dx = (w3d.xmmax-w3d.xmmin)/w3d.nx
w3d.dz = (w3d.zmmax-w3d.zmmin)/w3d.nz

w3d.solvergeom = w3d.XYZgeom
w3d.ymmin = Y_MIN
w3d.ymmax = Y_MAX
w3d.ny = NUM_Y
w3d.dy = (w3d.ymmax-w3d.ymmin)/w3d.ny


zmesh = np.linspace(0,Z_MAX,NUM_Z+1) #holds the z-axis grid points in an array
xmesh = np.linspace(X_MIN,X_MAX,NUM_X+1)

ANODE_VOLTAGE = 7.5
CATHODE_VOLTAGE = 0.
vacuum_level = ANODE_VOLTAGE - CATHODE_VOLTAGE
beam_beta = 5e-4
#Determine an appropriate time step based upon estimated final velocity
vzfinal = sqrt(2.*abs(vacuum_level)*np.abs(e)/m_e)+beam_beta*c
dt = w3d.dz/vzfinal
top.dt = 0.25*dt

if vzfinal*top.dt > w3d.dz:
    print "Time step dt = {:.3e}s does not constrain motion to a single cell".format(top.dt)

top.depos_order = 1
f3d.mgtol = 1e-9 # Multigrid solver convergence tolerance, in volts. 1 uV is default in Warp.


solverE = MRBlock3D() #Dielectric()

    
registersolver(solverE)
r_cone = 0.05e-6 #X_MAX/8.0
z_cone = r_cone #Z_MAX/5.0

MR_1 = 4
MR_2 = 2

#Add patches
solverE.addchild(mins=[-0.5e-6, -0.5e-6, 0], maxs=[0.5e-6, 0.5e-6, 0.4e-6], refinement = [2,2,2])
solverE.children[0].addchild(mins=[-0.2e-6, -0.2e-6, 0], maxs=[0.2e-6, 0.2e-6, 0.2e-6], refinement = [2,2,2])

#Define conductor/dielectrics

source = ZPlane(voltage=CATHODE_VOLTAGE, zcent=w3d.zmmin+0.*w3d.dz,zsign=-1.)
solverE.installconductor(source, dfill=largepos)

plate = ZPlane(voltage=ANODE_VOLTAGE, zcent=Z_MAX-0.*w3d.dz)
solverE.installconductor(plate, dfill=largepos)

#Define dielectric sphere centered in domain
r_sphere = Z_MAX/8.
epsn = 7.5 #dielectric constant for sphere

#define centered coordinates
X0 = 0.5*(w3d.xmmax + w3d.xmmin)
Y0 = 0.5*(w3d.ymmax + w3d.ymmin)
Z0 = 0.5*(w3d.zmmax + w3d.zmmin)


#Define diagnostics
particleperiod = 1
particle_diagnostic_0 = ParticleDiagnostic(period = particleperiod, top = top, w3d = w3d,
                                          species = {species.name: species for species in listofallspecies},
                                          comm_world=comm_world, lparallel_output=False, write_dir = diagDir[:-5])
fieldperiod = 1
efield_diagnostic_0 = ElectrostaticFieldDiagnostic(solver=solverE, top=top, w3d=w3d, comm_world = comm_world,
                                      period=fieldperiod, write_dir = diagFDir['electric'])

installafterstep(particle_diagnostic_0.write)
installafterstep(efield_diagnostic_0.write)

if comm_world.rank == 0:
    print("Conductors and diagnostics installed")

    
# --- Setup simulation species
#top.inject = 1
#top.vinject = 10.0
#top.zinject = 0.0
#top.npinject = 300

source_temperature = 2200

beam = Species(type = Electron, charge_state = 1, name = 'beam')
beam.a0 = CHANNEL_WIDTH
beam.b0 = CHANNEL_WIDTH
beam.vthz = sqrt(source_temperature*kb/beam.mass)
#beam.ibeam = 1000.


# --- Set the time step size. This needs to be small enough to satisfy the Courant limit.
#top.dt = 1.0e-14

#def injector():
#    z_bts = np.linspace(0, r_cone / 2, 5)    
#    x_bts = np.sqrt(r_cone ** 2 - (z_bts**2))
#    beam.addparticles(x = -0.1e-6 + x_bts, 
#                      y = 0 * x_bts, 
#                      z = z_bts, 
#                      vx = 0 * z_bts, 
#                      vy = 0 * z_bts, 
#                      vz = 10.0 + 0. * z_bts)
#installuserinjection(injector)

#Install sphere
e1 = emitters.conical(temperature = source_temperature, 
                x_offset = 0.0, y_offset = 0.0, voltage = 0.0, N_particles = 1000, output_to_file = True, file_name = 'emitted_particles_c7p5_%s.h5' % (int(source_temperature)))

e1.register(solverE, beam, top)
e1.install_conductor()

targetz_particles = ZCrossingParticles(zz=0.99 * Z_MAX,laccumulate=1)

#####################################
# Generate PIC code and Run Simulation
#####################################
solverE.mgmaxiters = 1

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

solverE.mgmaxiters = 16000 #rough approximation of steps needed for generate() to converge
solverE.children[0].mgmaxiters = 1000


package("w3d")


generate()


solverE.mgmaxiters = 500

#User injection
step()
installuserinjection(e1.injector)



step(5000)

x = beam.getx()
y = beam.gety()
z = beam.getz()
vx = beam.getvx()
vy = beam.getvy()
vz = beam.getvz()
w = beam.getw()

particles = np.column_stack([x, y, z, vx, vy, vz, w])

np.save('beam_dump_c_%s_%s.npy' % (int(ANODE_VOLTAGE), int(source_temperature)), particles)


x = np.zeros(100)
y = np.zeros(100)
z = np.linspace(0, Z_MAX, 100, endpoint=True)

Ex_tip = fzeros(x.shape)
Ey_tip = fzeros(x.shape)
Ez_tip = fzeros(x.shape)
Bx_tip = fzeros(x.shape)
By_tip = fzeros(x.shape)
Bz_tip = fzeros(x.shape)
phi_tip = fzeros(x.shape)

solverE.fetchfieldfrompositions(x, y, z, Ex_tip, Ey_tip, Ez_tip, Bx_tip, By_tip, Bz_tip)
solverE.fetchpotentialfrompositions(x, y, z, phi_tip)

np.save('axis_fields_c_%s_%s.npy' % (int(ANODE_VOLTAGE), int(source_temperature)), np.column_stack([x, y, z, Ex_tip, Ey_tip, Ez_tip, phi_tip, Bx_tip, By_tip, Bz_tip]))

x = targetz_particles.getx()
y = targetz_particles.gety()
vz = targetz_particles.getvz()
vx = targetz_particles.getvx()
vy = targetz_particles.getvy()
t = targetz_particles.gett()
w = targetz_particles.getpid()

particles = np.column_stack([x, y, t, vx, vy, vz, w])

np.save('target_particles_c_%s_%s.npy' % (int(ANODE_VOLTAGE), int(source_temperature)), particles)