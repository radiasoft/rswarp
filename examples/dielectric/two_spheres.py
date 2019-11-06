# # Test script simulating two intersecting spheres (one dielectric, one conducting) between two parallel plates.
#
# Updated for testing and validating cut-cells - Nathan Cook
#
# If l_MR is set, then two mesh refinement patches are applied over the sphere and their intersection.
#
# If USE_3D is set, then a 3D cartesian geometry is used rather than a 2D.
# 
# If MAKE_PLOTS is set, then plots a 2D slice of the potential and compares a 1D lineout to the analytic solution.
#
#
# 01/09/2018
#  
# Jean-Luc Vay

from __future__ import division
import sys
import os

import matplotlib as mpl
mpl.use('Agg')

from warp import * 
from warp.data_dumping.openpmd_diag import ParticleDiagnostic
from warp.data_dumping.openpmd_diag import ElectrostaticFieldDiagnostic

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5

# --- user flags
L_MR       = False #IF true, use mesh refinement
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


def get_block_potential(block,scale=1e6):
    '''Return the potential of a block to be plotted along with the imshow extent'''
    
    ngx,ngy,ngz = block.nxguardphi, block.nyguardphi, block.nzguardphi #phi guard cells -> look at nguarddepos for rho
    
    #Adjust between 3D and 2D slicing
    if ngx == 0 or ngy == 0 or ngz == 0:
        USE_3D = False
        #Assume we choose z,x for now.
        phi = block.potential[ngx:-ngx,:,ngz:-ngz]
    else:
        USE_3D = True
        phi = block.potential[ngx:-ngx,ngy:-ngy,ngz:-ngz]
        
    
    #Determine mesh properties
    zmesh = block.zmesh
    ymesh = block.ymesh
    xmesh = block.xmesh
    
    numx = xmesh.shape[0]
    numy = ymesh.shape[0]
    numz = zmesh.shape[0]
    
    x_mid = int(numx/2)
    y_mid = int(numy/2)
    
    xl = 0
    xu = numx-1
    zl = 0 
    zu = numz-1
    
    if USE_3D:
        plot_phi = phi[xl:xu+1,y_mid,zl:zu+1]
    else:
        plot_phi = phi[xl:xu+1,0,zl:zu+1]
    
    #Extent of the array to plot -> in this case plot the entire scaled domain
    pxmin = block.xmesh[xl] * scale
    pxmax = block.xmesh[xu] * scale
    pzmin = block.zmesh[zl] * scale
    pzmax = block.zmesh[zu] * scale
    
    plot_extent = [pzmin, pzmax, pxmin, pxmax]
    
    return plot_phi, plot_extent
                        
def plot_full_phi(block, show_grid=False, scale=1e6, svnm='phi', **kwargs):
    '''
    This function will recursively grab each block and the corresponding potential/mesh for plotting on the same figure.
    
    Arguments:
    block - The top level Warp solver, containing the blocklist and relevant children
    show_grid - If true, draw the grid for each block
    scale - Scaling of the dimensions of the domain being plotted. Defaults to 1e6.
    '''
    
    #verify that in case of mesh refinement, the primary block has been passed
    try: 
        #Test if Mesh Refinement solver is being used
        if (solverE.blocknumber is not None):
            USE_MR = True
            #The primary block (initial domain) has blocknumber 0
            assert block.blocknumber == 0, "Argument block must be primary block."
    except AttributeError:
        #No Mesh Refinement
        USE_MR = False
    
    #create initial figure
    fig = plt.figure(figsize=(18,9))
    ax = fig.add_subplot(1,1,1)
    
    ax.set_xlabel("z ($\mu$m)",fontsize=15)
    ax.set_ylabel("x ($\mu$m)",fontsize=15)
    ax.set_title(r"$\phi$ across domain",fontsize=18)
    
    #set initial scale
    ax.set_xlim(block.zmmin*scale, block.zmmax*scale)
    ax.set_ylim(block.xmmin*scale, block.xmmax*scale)
    
    if USE_MR:
        #Loop through all blocks
        #Must layer plots, with main block plotted first
        for subblock in block.listofblocks:

            #Get phi and extent for a given block
            phi,extent = get_block_potential(subblock,scale)

            #Set color scale using master block
            if subblock.blocknumber==0:
                vmin = np.min(phi)
                vmax = np.max(phi)

            #Plot using imshow on the same axis
            phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax)

            #Add color bar for master block
            if subblock.blocknumber==0:    
                cbar = fig.colorbar(phi_plt)
                cbar.ax.set_xlabel(r"$\phi [V]$")

            if show_grid:
                zmeshx,xmeshz = np.meshgrid(subblock.zmesh,subblock.xmesh)
                gridzx = create_grid_lines(zmeshx*1e6,xmeshz*1e6,alpha=0.25,linestyle='dashed')
                ax.add_collection(gridzx)
    else:
        #Get phi and extent for the non-MR solver
        phi,extent = get_block_potential(solverE,scale)
        
        #Set color scale
        vmin = np.min(phi)
        vmax = np.max(phi)

        #Plot using imshow on the same axis
        phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax)
        
        #Add the color bar
        cbar = fig.colorbar(phi_plt)
        cbar.ax.set_xlabel(r"$\phi [V]$")

        if show_grid:
            zmeshx,xmeshz = np.meshgrid(solverE.zmesh,solverE.xmesh)
            gridzx = create_grid_lines(zmeshx*1e6,xmeshz*1e6,alpha=0.25,linestyle='dashed')
            ax.add_collection(gridzx)
    
    
    if show_grid:
        #svnm = svnm[:-4] + '_grid' + string[-4:]
        svnm+='_grid'
        
    fig.savefig(svnm + '.png')
    
def create_grid_lines(X,Y,rstride = 1, cstride = 1, *args,**kwargs):
    '''Construct line collection for plotting a slice of the mesh used during the solve'''
    
    X, Y = np.broadcast_arrays(X, Y)
    
    # We want two sets of lines, one running along the "rows" of
    # Z and another set of lines running along the "columns" of Z.
    # This transpose will make it easy to obtain the columns.
    tX, tY = np.transpose(X), np.transpose(Y)
    
    rows, cols = rows, cols = X.shape
    
    if rstride:
        rii = list(range(0, rows, rstride))
        # Add the last index only if needed
        if rows > 0 and rii[-1] != (rows - 1):
            rii += [rows-1]
    else:
        rii = []
    if cstride:
        cii = list(range(0, cols, cstride))
        # Add the last index only if needed
        if cols > 0 and cii[-1] != (cols - 1):
            cii += [cols-1]
    else:
        cii = []

    xlines = [X[i] for i in rii]
    ylines = [Y[i] for i in rii]
    #zlines = [Z[i] for i in rii]

    txlines = [tX[i] for i in cii]
    tylines = [tY[i] for i in cii]
    #tzlines = [tZ[i] for i in cii]

    lines = ([list(zip(xl, yl))
              for xl, yl in zip(xlines, ylines)]
            + [list(zip(xl, yl))
               for xl, yl in zip(txlines, tylines)])
    
    line_collection = mpl.collections.LineCollection(lines, *args, **kwargs)
    
    return line_collection
    

############################
# Set User parmeters       #
############################

# Constants imports
from scipy.constants import e, m_e, c, k
kb_eV = 8.6173324e-5 #Bolztmann constant in eV/K
kb_J = k #Boltzmann constant in J/K
m = m_e

diagDir = 'diags/xzsolver/hdf5/'
field_base_path = 'diags/fields/'
diagFDir = {'magnetic':'diags/fields/magnetic','electric':'diags/fields/electric'}

# Cleanup previous files
if comm_world.rank == 0:
    cleanupPrevious(diagDir,diagFDir)

if comm_world.size != 1:
    synchronizeQueuedOutput_mpi4py(out=False, error=False)

top.inject = 0 
top.npinject = 0

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
if comm_world.size > 1:
    N_ALL = 95
else:
    if L_MR == True:
        N_ALL = 32
    else:
        N_ALL = 64

NUM_X = N_ALL
NUM_Y = N_ALL
NUM_Z = N_ALL


## Solver Geometry

# Set boundary conditions
w3d.bound0  = dirichlet
w3d.boundnz = dirichlet
w3d.boundxy = periodic 

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

if USE_3D:
    w3d.solvergeom = w3d.XYZgeom
    w3d.ymmin = Y_MIN
    w3d.ymmax = Y_MAX
    w3d.ny = NUM_Y
    w3d.dy = (w3d.ymmax-w3d.ymmin)/w3d.ny
else:
    w3d.solvergeom = w3d.XZgeom

zmesh = np.linspace(0,Z_MAX,NUM_Z+1) #holds the z-axis grid points in an array
xmesh = np.linspace(X_MIN,X_MAX,NUM_X+1)

ANODE_VOLTAGE = 10.
CATHODE_VOLTAGE = 0.
vacuum_level = ANODE_VOLTAGE - CATHODE_VOLTAGE
beam_beta = 5e-4
#Determine an appropriate time step based upon estimated final velocity
vzfinal = sqrt(2.*abs(vacuum_level)*np.abs(e)/m_e)+beam_beta*c
dt = w3d.dz/vzfinal
top.dt = 0.5*dt

if vzfinal*top.dt > w3d.dz:
    print "Time step dt = {:.3e}s does not constrain motion to a single cell".format(top.dt)

top.depos_order = 1
f3d.mgtol = 1e-6 # Multigrid solver convergence tolerance, in volts. 1 uV is default in Warp.

if USE_3D:
    if L_MR:
        solverE = MRBlock3DDielectric()
    else:
        solverE = MultiGrid3DDielectric()
else:
    if L_MR:
        solverE = MRBlock2DDielectric()
    else:
        solverE = MultiGrid2DDielectric()
    
registersolver(solverE)

#Add patches
if L_MR:
    solverE.addchild(mins=[-0.25e-6,-0.25e-6,0.25e-6],maxs=[0.25e-6,0.25e-6,0.75e-6])
    solverE.children[0].addchild(mins=[-0.2e-6,-0.2e-6,0.5e-6],maxs=[0.2e-6,0.2e-6,0.7e-6])

#Define conductor/dielectrics

source = ZPlane(voltage=CATHODE_VOLTAGE, zcent=w3d.zmmin+0.*w3d.dz,zsign=-1.)
solverE.installconductor(source, dfill=largepos)

plate = ZPlane(voltage=ANODE_VOLTAGE, zcent=Z_MAX-0.*w3d.dz)
solverE.installconductor(plate,dfill=largepos)

#Define dielectric sphere
r_sphere = Z_MAX/8.
Z0 = 0.5e-6
epsn = 7.5 #dielectric constant for sphere

sphere = Sphere(radius=r_sphere, xcent=0.0, ycent=0.0, zcent=Z0, permittivity=epsn)
solverE.installconductor(sphere,dfill=largepos)

#Installs an offset conducting sphere
spherecond = Sphere(radius=1.*r_sphere, xcent=0.5*w3d.xmmin, ycent=0.0, zcent=Z0, voltage=ANODE_VOLTAGE/10)
solverE.installconductor(spherecond,dfill=largepos)

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

# Generate PIC code and Run Simulation
solverE.mgmaxiters = 1

#prevent GIST from starting upon setup
top.lprntpara = false
top.lpsplots = false
top.verbosity = 0 

solverE.mgmaxiters = 16000 #rough approximation of steps needed for generate() to converge
if L_MR:
    solverE.children[0].mgmaxiters = 1000
    solverE.children[0].children[0].mgmaxiters = 1000

package("w3d")
generate()
solverE.mgmaxiters = 100
step(1)



################
#NOW PLOT STUFF
################

if (comm_world.rank == 0 and MAKE_PLOTS):
    
    #Plot a 2D slice of the potential
    if USE_3D:
        if L_MR:
            svtitle = 'phi_mr-{}x{}x{}'.format(NUM_X,NUM_Y,NUM_Z)
        else:
            svtitle = 'phi-{}x{}x{}'.format(NUM_X,NUM_Y,NUM_Z)
    else:
        if L_MR:
            svtitle = 'phi_mr-{}x{}'.format(NUM_X,NUM_Z)
        else:
            svtitle = 'phi-{}x{}'.format(NUM_X,NUM_Z) 
    
    
    plot_full_phi(solverE,show_grid=True, svnm=svtitle)
    
    
    #Calculate Theoretical Potential
    d = 1e-6
    a = d/8.
    phi_a = 10. # 10 V

    e_r = 7.5

    [Z,X] = np.meshgrid(zmesh+1e-12,xmesh+1e-12)

    Z0 = 0.5e-6

    R = np.sqrt((Z-Z0)**2 + X**2)

    inside = (R <= a)

    phicalc = phi_a *(Z-Z0) /d *(1.-(e_r-1.)/(e_r+2.)*(a/R)**3.)
    phicalc[inside] = 3./(e_r+2.)*phi_a *(Z[inside]-Z0)/d
    
    #Grab a lineout of the potential
    phi = solverE.getphi()
    x_mid = int(NUM_X/2)
    y_mid = int(NUM_Y/2)
    phisim = fzeros(R[0,:].shape)
    solverE.fetchpotentialfrompositions(X[x_mid,:],0.*X[x_mid,:],Z[x_mid,:],phisim)

    #comparison
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    if USE_3D:
        ax.plot(zmesh*1e6,phicalc[x_mid,:]+5., label = 'No conducting sphere')
        ax.plot(zmesh*1e6,phi[x_mid,y_mid,:], label = 'Simulated (coarse grid only)')
    else:
        ax.plot(zmesh*1e6,phicalc[x_mid,:]+5., label = 'No conducting sphere')
        ax.plot(zmesh*1e6,phi[x_mid,:], label = 'Simulated (coarse grid only)')
    if L_MR:
        ax.plot(zmesh*1e6,phisim[:], label = 'Simulated (coarse+MR)')
    ax.set_title('Potential through dielectric sphere')
    ax.set_xlabel('z [$\mu$m]')
    ax.set_ylabel('$\phi$ [V]')
    plt.legend()
    
    if USE_3D:
        if L_MR:
            svtitle = 'twospheres_phi_lineout_mr-{}x{}x{}.png'.format(NUM_X,NUM_Y,NUM_Z)
        else:
            svtitle = 'twospheres_phi_lineout-{}x{}x{}.png'.format(NUM_X,NUM_Y,NUM_Z)
    else:
        if L_MR:
            svtitle = 'twospheres_phi_lineout_mr-{}x{}.png'.format(NUM_X,NUM_Z)
        else:
            svtitle = 'twospheres_phi_lineout-{}x{}.png'.format(NUM_X,NUM_Z)        
    
    fig.savefig(svtitle,bbox_inches='tight')
    
    
    #Calculate Theoretical Potential
    d = 1e-6
    a = d/8.
    phi_a = 10. # 10 V

    e_r = 7.5

    [Z,X] = np.meshgrid(zmesh+1e-12,xmesh+1e-12)

    Z0 = 0.5e-6

    R = np.sqrt((Z-Z0)**2 + X**2)

    inside = (R <= a)

    phicalc = phi_a *(Z-Z0) /d *(1.-(e_r-1.)/(e_r+2.)*(a/R)**3.)
    phicalc[inside] = 3./(e_r+2.)*phi_a *(Z[inside]-Z0)/d
    
    #Grab a lineout of the potential
    phi = solverE.getphi()
    x_mid = int(NUM_X/2)
    y_mid = int(NUM_Y/2)
    z_mid = int(NUM_Z/2)

    phisimz = fzeros(R[0,:].shape)
    solverE.fetchpotentialfrompositions(X[:,z_mid],0.*X[:,z_mid],Z[:,z_mid],phisimz)
    
    #comparison
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    if USE_3D:
        #ax.plot(zmesh*1e6,phicalc[:,z_mid]+5., label = 'No conducting sphere')
        ax.plot(zmesh*1e6,phi[:,y_mid,z_mid], label = 'Simulated (coarse grid only)')
    else:
        #ax.plot(zmesh*1e6,phicalc[:,z_mid]+5., label = 'No conducting sphere')
        ax.plot(zmesh*1e6,phi[:,z_mid], label = 'Simulated (coarse grid only)')
    if L_MR:
        ax.plot(zmesh*1e6,phisimz[:], label = 'Simulated (coarse+MR)')
    ax.set_title('Potential across intersection of conducting and dielectric spheres')
    ax.set_xlabel('x [$\mu$m]')
    ax.set_ylabel('$\phi$ [V]')
    plt.legend()
    
    if USE_3D:
        if L_MR:
            svtitle = 'twospheres_intersection_mr-{}x{}x{}.png'.format(NUM_X,NUM_Y,NUM_Z)
        else:
            svtitle = 'twospheres_intersection_lineout-{}x{}x{}.png'.format(NUM_X,NUM_Y,NUM_Z)
    else:
        if L_MR:
            svtitle = 'twospheres_intersection_lineout_mr-{}x{}.png'.format(NUM_X,NUM_Z)
        else:
            svtitle = 'twospheres_intersection_lineout-{}x{}.png'.format(NUM_X,NUM_Z)        
    
    fig.savefig(svtitle,bbox_inches='tight')