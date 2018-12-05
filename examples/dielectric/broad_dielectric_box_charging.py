# # Dielectric charging script development and testing
# 
# **7/18/2017**
# 
# This notebook will develop and troubleshoot scripts for testing dielectric charging from electron impact.
# 
# 1. Simple geometry (fast) - 1 micron x 1 micron (z). 100 x 100 grid
# 2. Rectangular dielectric - 500 nm x 100 nm (z). Centered transversely, closer to anode along z.
# 3. Uniform parallel emission -> 20 particles, user-injected,no transverse velocity, clustered about central axis.
# 
# Nathan Cook

from __future__ import division
import sys
del sys.argv[1:] #  Necessry to run 'from warp import *' in IPython notebook without conflict.
from warp import * 
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import os
import pickle
import h5py
from re import findall
from scipy.special import erfinv
from datetime import datetime

import rswarp
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from rswarp.diagnostics import FieldDiagnostic
from rswarp.utilities.file_utils import cleanupPrevious
from rswarp.utilities.file_utils import readparticles
from rswarp.utilities.file_utils import loadparticlefiles
from rswarp.cathode import sources
from rswarp.cathode import injectors
from warp.particles.singleparticle import TraceParticle

import matplotlib.lines as mlines
import matplotlib.patches as patches

# Constants imports
from scipy.constants import e, m_e, c, k
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e


diagDir = 'diags/xzsolver/hdf5/'
field_base_path = 'diags/fields/'
diagFDir = {'magnetic':'diags/fields/magnetic','electric':'diags/fields/electric'}

# Cleanup previous files
cleanupPrevious(diagDir,diagFDir)


### Grid parameters, Solver, and Boundaries
epsn = 7.

if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=False, error=False)

print "rank:", comm_world.rank

top.inject = 0 
top.npinject = 0

#Dimensions

PLATE_SPACING = 1.e-6 #plate spacing
CHANNEL_WIDTH = 1e-6 #width of simulation box

X_MAX = CHANNEL_WIDTH*0.5
X_MIN = -1.*X_MAX
Y_MAX = CHANNEL_WIDTH*0.5
Y_MIN = -1.*Y_MAX
Z_MIN = 0.
Z_MAX = PLATE_SPACING


#Grid parameters
NUM_X = 100 #256-1#64-1
NUM_Z = 100 #256-1#64-1

# # Solver Geometry

w3d.solvergeom = w3d.XZgeom


# Set boundary conditions
w3d.bound0  = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = periodic 


# Set grid boundaries
w3d.xmmin = X_MIN
w3d.xmmax = X_MAX
w3d.zmmin = 0. 
w3d.zmmax = Z_MAX

w3d.nx = NUM_X
w3d.nz = NUM_Z

w3d.dx = (w3d.xmmax-w3d.xmmin)/w3d.nx
w3d.dz = (w3d.zmmax-w3d.zmmin)/w3d.nz

zmesh = np.linspace(0,Z_MAX,NUM_Z+1) #holds the z-axis grid points in an array

ANODE_VOLTAGE = 10.
CATHODE_VOLTAGE = 0.
vacuum_level = ANODE_VOLTAGE - CATHODE_VOLTAGE
beam_beta = 5e-4
#Determine an appropriate time step based upon estimated final velocity
vzfinal = sqrt(2.*abs(vacuum_level)*np.abs(e)/m_e)+beam_beta*c
dt = w3d.dz/vzfinal #5e-15
top.dt = 0.1*dt

if vzfinal*top.dt > w3d.dz:
    print "Time step dt = {:.3e}s does not constrain motion to a single cell".format(top.dt)


#### Set up field solver

top.depos_order = 1
f3d.mgtol = 1e-6 # Multigrid solver convergence tolerance, in volts. 1 uV is default in Warp.
solverE = MultiGrid2DDielectric()
registersolver(solverE)


#### Define conductors and dielectrics using new wrapper

source = ZPlane(zcent=w3d.zmmin+0*w3d.dz,zsign=-1.,voltage=CATHODE_VOLTAGE)
solverE.installconductor(source, dfill=largepos)

plate = ZPlane(voltage=ANODE_VOLTAGE, zcent=Z_MAX-0.*w3d.dz)
solverE.installconductor(plate,dfill=largepos)


box = Box(xsize=0.5*(w3d.xmmax-w3d.xmmin),
          ysize=0.5*(w3d.ymmax-w3d.ymmin),
          zsize=0.1*(w3d.zmmax-w3d.zmmin),
          xcent=0.5*(w3d.xmmax+w3d.xmmin),
          ycent=0.5*(w3d.ymmax+w3d.ymmin),
          zcent=0.8*(w3d.zmmax+w3d.zmmin),
          permittivity=epsn)

solverE.installconductor(box,dfill=largepos)


### Diagnostics

particleperiod = 100
particle_diagnostic_0 = ParticleDiagnostic(period = particleperiod, top = top, w3d = w3d,
                                          species = {species.name: species for species in listofallspecies},
                                          comm_world=comm_world, lparallel_output=False, write_dir = diagDir[:-5])
fieldperiod = 100
efield_diagnostic_0 = FieldDiagnostic.ElectrostaticFields(solver=solverE, top=top, w3d=w3d, comm_world = comm_world,
                                                          period=fieldperiod)

installafterstep(particle_diagnostic_0.write)
installafterstep(efield_diagnostic_0.write)


### Generate and Run

#Generate PIC code and Run Simulation
solverE.mgmaxiters = 1

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

solverE.mgmaxiters = 10000 #rough approximation needed for initial solve to converge
package("w3d")
generate()
solverE.mgmaxiters = 100


#Need to compute the fields first
epsilon_array = solverE.epsilon/eps0

#Now plot
fig = plt.figure(figsize=(12,6))

X_CELLS = NUM_X
Z_CELLS = NUM_Z

xl = 0
xu = NUM_X
zl = 0 
zu = NUM_Z 

plt.xlabel("z ($\mu$m)")
plt.ylabel("x ($\mu$m)")
plt.title(r"$\kappa$ across domain - with box")

pxmin = ((X_MAX - X_MIN) / X_CELLS * xl + X_MIN) * 1e6
pxmax = ((X_MAX - X_MIN) / X_CELLS * xu + X_MIN) * 1e6
pzmin = (Z_MIN + zl / Z_CELLS * Z_MAX) * 1e6
pzmax = (Z_MAX * zu / Z_CELLS) * 1e6

plt.xlim(pzmin, pzmax)
plt.ylim(pxmin, pxmax)

eps_plt = plt.imshow(epsilon_array[xl:xu,zl:zu],cmap='viridis',extent=[pzmin, pzmax, pxmin, pxmax],aspect='auto')

cbar = fig.colorbar(eps_plt)
cbar.ax.set_xlabel(r"$\kappa$")
cbar.ax.xaxis.set_label_position('top')

plt.savefig('eps_broad_box.png',bbox_inches='tight')


#### Specify emission

electrons_tracked_t0 = Species(type=Electron, weight=1.0)
ntrack = 20
Z_PART_MIN = w3d.dz/8 #Add a minimum z coordinate to prevent absorption

# Uniform velocity used for all particles
x_vals = np.arange(-0.25*CHANNEL_WIDTH,0.25*CHANNEL_WIDTH,0.5*CHANNEL_WIDTH / ntrack)
y_vals = CHANNEL_WIDTH*(np.random.rand(ntrack)-0.5)
z_vals = np.zeros(ntrack) + Z_PART_MIN #Add a minimum z coordinate to prevent absorption

vx_vals = np.zeros(ntrack)
vy_vals = np.zeros(ntrack)
vz_vals = beam_beta*clight*np.ones(ntrack) #beta = 0.0005

eptclArray = np.asarray([x_vals,vx_vals,y_vals,vy_vals,z_vals,vz_vals]).T

electron_tracker_0 = TraceParticle(js=electrons_tracked_t0.jslist[0],
                     x=x_vals,
                     y=y_vals,
                     z=z_vals,
                     vx=vx_vals,
                     vy=vy_vals,
                     vz=vz_vals)


num_steps = 2000
step(num_steps)

print solverE.getselfe().shape
print getselfe('z').shape
zfield = getselfe('z')
if comm_world.size > 1:
	if comm_world.rank == 0:
		np.save('diel_para.npy',zfield)
elif comm_world.size == 1:
	np.save('diel_ser.npy',zfield)



### Plot particle trajectories

def particle_trace(trace,ntrack):
    kept_electronsx = []
    kept_electronsz = []
    lost_electronsx = []
    lost_electronsz = []
    for electron in range(ntrack):
        for step in range(len(trace.getx(i=electron)) - 1):
            if abs(trace.getx(i=electron)[step] - 
                   trace.getx(i=electron)[step + 1]) > (X_MAX - X_MIN) / 2.:
                lost_electronsx.append(trace.getx(i=electron)[0:step])
                lost_electronsz.append(trace.getz(i=electron)[0:step])
                break
            if step == (len(trace.getx(i=electron)) - 2):
                kept_electronsx.append(trace.getx(i=electron))
                kept_electronsz.append(trace.getz(i=electron))
    return [kept_electronsx,kept_electronsz], [lost_electronsx,lost_electronsz]

kept_electrons, lost_electrons = particle_trace(electron_tracker_0,ntrack)

cond_list = solverE.conductordatalist

fig = plt.figure(figsize=(12,6))
plt.title("Broad Dielectric Particle Trace")

scale = 1e6

ax2 = plt.subplot(111)

steps2cross = 1900 #computed # of steps to cross

cond_list = solverE.conductordatalist
for cond in cond_list[2:]: #ignore first two conductors - these are the plates
    co = cond[0]
    specs = co.getkwlist()
    xw = specs[0] #x-dimension (y in plot)
    xc = specs[3] #center
    xll = xc - xw/2. #lower left
    zw = specs[2] #z-dimension (x in plot)
    zc = specs[-1] #center
    zll = zc - zw/2. #lower left

    
    ax2.add_patch(
        patches.Rectangle(
            (zll * scale, xll * scale),
            zw * scale,
            xw * scale,
            facecolor="grey",
            edgecolor="grey"  
        )
    )

ax1 = plt.subplot(111)
kept_electrons, lost_electrons = particle_trace(electron_tracker_0,ntrack)

for i in range(len(kept_electrons[1])):
    ax1.plot(kept_electrons[1][i][:steps2cross] * scale,kept_electrons[0][i][:steps2cross] * scale, c = '#1f77b4')

for i in range(len(lost_electrons[1])):
    ax1.plot(lost_electrons[1][i][:steps2cross] * scale,lost_electrons[0][i][:steps2cross] * scale, c = '#2ca02c')


kept = mlines.Line2D([], [], color='#1f77b4',label='Absorbed Particles')
lost = mlines.Line2D([], [], color='#2ca02c',label='Reflected Particles')

box = ax1.get_position()
ax1.set_position([box.x0, box.y0, box.width * 0.8, box.height])

plt.xlim(Z_MIN * scale,Z_MAX * scale)
plt.ylim(X_MIN * scale, X_MAX * scale)
plt.legend(handles=[kept, lost],loc='best', bbox_to_anchor=(1, 1))
plt.xlabel('z ($\mu$m)')
plt.ylabel('x ($\mu$m)')
plt.savefig('broad_dielectric_trace.png')
plt.show()



### Plot fields

#Need to compute the fields first
fieldEz = solverE.getez()

#Now plot
fig = plt.figure(figsize=(12,6))

X_CELLS = NUM_X
Z_CELLS = NUM_Z

xl = 0
xu = NUM_X
zl = 0 
zu = NUM_Z 

plt.xlabel("z ($\mu$m)")
plt.ylabel("x ($\mu$m)")
plt.title("$E_z$ across domain - with box")

pxmin = ((X_MAX - X_MIN) / X_CELLS * xl + X_MIN) * 1e6
pxmax = ((X_MAX - X_MIN) / X_CELLS * xu + X_MIN) * 1e6
pzmin = (Z_MIN + zl / Z_CELLS * Z_MAX) * 1e6
pzmax = (Z_MAX * zu / Z_CELLS) * 1e6

plt.xlim(pzmin, pzmax)
plt.ylim(pxmin, pxmax)

ez_plt = plt.imshow(fieldEz[xl:xu,zl:zu],cmap='viridis',extent=[pzmin, pzmax, pxmin, pxmax],aspect='auto')

cbar = fig.colorbar(ez_plt)
cbar.ax.set_xlabel("V/m")
cbar.ax.xaxis.set_label_position('top')

plt.savefig('Ez_broad_box.png',bbox_inches='tight')
plt.close()

#Need to compute the fields first
fieldEx = solverE.getex()

Exr = fieldEx[::-1]

#Now plot
fig = plt.figure(figsize=(12,6))

X_CELLS = NUM_X
Z_CELLS = NUM_Z

xl = 0
xu = NUM_X
zl = 0 
zu = NUM_Z 

plt.xlabel("z ($\mu$m)")
plt.ylabel("x ($\mu$m)")
plt.title("$E_x$ across domain - with box")

pxmin = ((X_MAX - X_MIN) / X_CELLS * xl + X_MIN) * 1e6
pxmax = ((X_MAX - X_MIN) / X_CELLS * xu + X_MIN) * 1e6
pzmin = (Z_MIN + zl / Z_CELLS * Z_MAX) * 1e6
pzmax = (Z_MAX * zu / Z_CELLS) * 1e6

plt.xlim(pzmin, pzmax)
plt.ylim(pxmin, pxmax)

ex_plt = plt.imshow(Exr[xl:xu,zl:zu],cmap='viridis',extent=[pzmin, pzmax, pxmin, pxmax],aspect='auto')


cbar = fig.colorbar(ex_plt)
cbar.ax.set_xlabel("V/m")
cbar.ax.xaxis.set_label_position('top')

plt.savefig('Ex_broad_box.png',bbox_inches='tight')
plt.close()


#### Plot potential

#Need to compute the potential first
potential = solverE.getphi()

#Now plot
fig = plt.figure(figsize=(12,6))

X_CELLS = NUM_X
Z_CELLS = NUM_Z

potential = solverE.getphi()

xl = 0
xu = NUM_X
zl = 0 
zu = NUM_Z 

plt.xlabel("z ($\mu$m)")
plt.ylabel("x ($\mu$m)")
plt.title("$\phi$ across domain -with box")

pxmin = ((X_MAX - X_MIN) / X_CELLS * xl + X_MIN) * 1e6
pxmax = ((X_MAX - X_MIN) / X_CELLS * xu + X_MIN) * 1e6
pzmin = (Z_MIN + zl / Z_CELLS * Z_MAX) * 1e6
pzmax = (Z_MAX * zu / Z_CELLS) * 1e6

plt.xlim(pzmin, pzmax)
plt.ylim(pxmin, pxmax)

phi_plt = plt.imshow(potential[xl:xu,zl:zu],cmap='RdBu',extent=[pzmin, pzmax, pxmin, pxmax],aspect='auto')

cbar = fig.colorbar(phi_plt)
cbar.ax.set_xlabel("Volts")
cbar.ax.xaxis.set_label_position('top')

plt.savefig('phi_broad_box.png',bbox_inches='tight')
plt.close()

