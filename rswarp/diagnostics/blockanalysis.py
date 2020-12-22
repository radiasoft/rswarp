import warpoptions
warpoptions.ignoreUnknownArgs = True

import matplotlib as mpl
import numpy as np
import warp
import matplotlib.pyplot as plt

def get_block_potential(block,scale=1e6):
    '''Return the potential of a block to be plotted along with the imshow extent in the X-Z plane'''

    ngx,ngy,ngz = block.nxguardphi, block.nyguardphi, block.nzguardphi #phi guard cells -> look at nguarddepos for rho

    #Adjust between 3D and 2D slicing
    if ngx == 0 or ngy == 0 or ngz == 0:
        USE_3D = False
        #print "2D solver"
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


def get_block_xy_potential(block, scale=1e6, zloc=None):
    '''Return the potential of a block to be plotted along with the imshow extent in the X-Y plane, if applicable'''

    ngx,ngy,ngz = block.nxguardphi, block.nyguardphi, block.nzguardphi #phi guard cells -> look at nguarddepos for rho

    #Adjust between 3D and 2D slicing
    if ngx == 0 or ngy == 0 or ngz == 0:
        USE_3D = False
        #Assume we choose x,z geometry for now.
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
    z_mid = int(numz/2)

    if zloc:
        zgrid = zloc
    else:
        zgrid = z_mid

    xl = 0
    xu = numx-1
    yl = 0
    yu = numy-1
    zl = 0
    zu = numz-1

    if USE_3D:
        plot_phi = phi[xl:xu+1,yl:yu+1,zgrid]
    else:
        plot_phi = phi[xl:xu+1,0,zl:zu+1]

    #Extent of the array to plot -> in this case plot the entire scaled domain
    pxmin = block.xmesh[xl] * scale
    pxmax = block.xmesh[xu] * scale
    pymin = block.ymesh[yl] * scale
    pymax = block.ymesh[yu] * scale
    pzmin = block.zmesh[zl] * scale
    pzmax = block.zmesh[zu] * scale

    plot_extent = [pxmin, pymax, pymin, pymax]

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
        if (block.blocknumber is not None):
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
            phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax, origin='lower')

            #Add color bar for master block
            if subblock.blocknumber==0:
                cbar = fig.colorbar(phi_plt)
                cbar.ax.set_xlabel(r"$\phi [V]$")

            if show_grid:
                zmeshx,xmeshz = np.meshgrid(subblock.zmesh,subblock.xmesh)
                gridzx = create_grid_lines(zmeshx*scale,xmeshz*scale,alpha=0.25,linestyle='dashed')
                ax.add_collection(gridzx)
    else:
        #Get phi and extent for the non-MR solver
        phi,extent = get_block_potential(block,scale)

        #Set color scale
        vmin = np.min(phi)
        vmax = np.max(phi)

        #Plot using imshow on the same axis
        phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax, origin='lower')

        #Add the color bar
        cbar = fig.colorbar(phi_plt)
        cbar.ax.set_xlabel(r"$\phi [V]$")

        if show_grid:
            zmeshx,xmeshz = np.meshgrid(block.zmesh,block.xmesh)
            gridzx = create_grid_lines(zmeshx*scale,xmeshz*scale,alpha=0.25,linestyle='dashed')
            ax.add_collection(gridzx)


    if show_grid:
        #svnm = svnm[:-4] + '_grid' + string[-4:]
        svnm+='_grid'

    fig.savefig(svnm + '.png')

def plot_xy_phi(block, show_grid=False, scale=1e6, z_loc=None, svnm='phi', **kwargs):
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
        if (block.blocknumber is not None):
            USE_MR = True
            #The primary block (initial domain) has blocknumber 0
            assert block.blocknumber == 0, "Argument block must be primary block."
    except AttributeError:
        #No Mesh Refinement
        USE_MR = False

    #create initial figure
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel("x ($\mu$m)",fontsize=15)
    ax.set_ylabel("y ($\mu$m)",fontsize=15)
    ax.set_title(r"$\phi$ across domain",fontsize=18)

    #set initial scale
    ax.set_xlim(block.xmmin*scale, block.xmmax*scale)
    ax.set_ylim(block.ymmin*scale, block.ymmax*scale)

    if USE_MR:
        #Loop through all blocks
        #Must layer plots, with main block plotted first
        for subblock in block.listofblocks:

            #Get phi and extent for a given block
            phi,extent = get_block_xy_potential(subblock,scale, zloc=z_loc)

            #Set color scale using master block
            if subblock.blocknumber==0:
                vmin = np.min(phi)
                vmax = np.max(phi)

            #Plot using imshow on the same axis
            phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax, origin='lower')

            #Add color bar for master block
            if subblock.blocknumber==0:
                cbar = fig.colorbar(phi_plt)
                cbar.ax.set_xlabel(r"$\phi [V]$")

            if show_grid:
                xmeshx,ymeshz = np.meshgrid(subblock.xmesh,subblock.ymesh)
                gridxy = create_grid_lines(xmeshx*scale,ymeshz*scale,alpha=0.25,linestyle='dashed')
                ax.add_collection(gridxy)
    else:
        #Get phi and extent for the non-MR solver
        phi,extent = get_block_xy_potential(block,scale, zloc=z_loc)

        #Set color scale
        vmin = np.min(phi)
        vmax = np.max(phi)

        #Plot using imshow on the same axis
        phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax, origin='lower')

        #Add the color bar
        cbar = fig.colorbar(phi_plt)
        cbar.ax.set_xlabel(r"$\phi [V]$")

        if show_grid:
            xmeshx,ymeshz = np.meshgrid(block.xmesh,block.ymesh)
            gridxy = create_grid_lines(xmeshx*scale,ymeshz*scale,alpha=0.25,linestyle='dashed')
            ax.add_collection(gridxy)


    if show_grid:
        #svnm = svnm[:-4] + '_grid' + string[-4:]
        svnm+='_grid'

    fig.savefig(svnm + '.png')



def plot_cathode_phi(block, show_grid=False, scale=1e6, z_loc=None, svnm='field', **kwargs):
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
        if (block.blocknumber is not None):
            USE_MR = True
            #The primary block (initial domain) has blocknumber 0
            assert block.blocknumber == 0, "Argument block must be primary block."
    except AttributeError:
        #No Mesh Refinement
        USE_MR = False

    #create initial figure
    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(1,1,1)

    ax.set_xlabel("x ($\mu$m)",fontsize=15)
    ax.set_ylabel("y ($\mu$m)",fontsize=15)
    ax.set_title(r"$\phi$ across domain",fontsize=18)

    #set initial scale
    ax.set_xlim(block.xmmin*scale, block.xmmax*scale)
    ax.set_ylim(block.ymmin*scale, block.ymmax*scale)

    if USE_MR:
        #Loop through all blocks
        #Must layer plots, with main block plotted first
        for subblock in block.listofblocks:

            #Get phi and extent for a given block
            phi,extent = get_block_xy_potential(subblock,scale, zloc=z_loc)

            #Set color scale using master block
            if subblock.blocknumber==0:
                vmin = np.min(phi)
                vmax = np.max(phi)

            #Plot using imshow on the same axis
            phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax, origin='lower')

            #Add color bar for master block
            if subblock.blocknumber==0:
                cbar = fig.colorbar(phi_plt)
                cbar.ax.set_xlabel(r"$\phi [V]$")

            if show_grid:
                xmeshx,ymeshz = np.meshgrid(subblock.xmesh,subblock.ymesh)
                gridxy = create_grid_lines(xmeshx*scale,ymeshz*scale,alpha=0.25,linestyle='dashed')
                ax.add_collection(gridxy)
    else:
        #Get phi and extent for the non-MR solver
        phi,extent = get_block_xy_potential(block,scale, zloc=z_loc)

        #Set color scale
        vmin = np.min(phi)
        vmax = np.max(phi)

        #Plot using imshow on the same axis
        phi_plt = ax.imshow(phi,cmap='viridis',extent=extent,aspect='auto',vmin=vmin, vmax=vmax, origin='lower')

        #Add the color bar
        cbar = fig.colorbar(phi_plt)
        cbar.ax.set_xlabel(r"$\phi [V]$")

        if show_grid:
            xmeshx,ymeshz = np.meshgrid(block.xmesh,block.ymesh)
            gridxy = create_grid_lines(xmeshx*scale,ymeshz*scale,alpha=0.25,linestyle='dashed')
            ax.add_collection(gridxy)


    if show_grid:
        #svnm = svnm[:-4] + '_grid' + string[-4:]
        svnm+='_grid'

    fig.savefig(svnm + '.png')

def get_block_fields(block, scale=1e6, use_3d=True):
    '''
    Return the electric fields of a block to be plotted along with the imshow extent in the X-Z plane.

    Returns the tuple of arrays (Ex, Ey, Ez, extent).
    '''

    #Define mesh on which to calculate fields
    xm, ym, zm = np.meshgrid(block.xmesh, block.ymesh, block.zmesh, indexing='ij')

    #Flatten for fetching
    xmf = xm.flatten()
    ymf = zm.flatten()
    zmf = zm.flatten()

    #Define flattened arrays for in-place fetch
    Exf = warp.fzeros(xmf.shape)
    Eyf = warp.fzeros(ymf.shape)
    Ezf = warp.fzeros(zmf.shape)
    Bxf = warp.fzeros(xmf.shape)
    Byf = warp.fzeros(ymf.shape)
    Bzf = warp.fzeros(zmf.shape)

    #Field fields
    block.fetchfieldfrompositions(xmf, ymf, zmf, Exf, Eyf, Ezf, Bxf, Byf, Bzf)

    #Now reshape fields for plotting
    Ex = Exf.reshape(xm.shape)
    Ey = Eyf.reshape(xm.shape)
    Ez = Ezf.reshape(xm.shape)

    #determine indexing for slicing
    numx = xm.shape[0]
    numy = ym.shape[0]
    numz = zm.shape[0]

    x_mid = int(numx/2)
    y_mid = int(numy/2)

    xl = 0
    xu = numx-1
    zl = 0
    zu = numz-1

    #Extent of the array to plot -> in this case plot the entire scaled domain
    pxmin = block.xmesh[xl] * scale
    pxmax = block.xmesh[xu] * scale
    pzmin = block.zmesh[zl] * scale
    pzmax = block.zmesh[zu] * scale

    plot_extent = [pzmin, pzmax, pxmin, pxmax]

    if use_3d:
        return Ex[xl:xu+1,y_mid,zl:zu+1], Ey[xl:xu+1,y_mid,zl:zu+1], Ez[xl:xu+1,y_mid,zl:zu+1], plot_extent

    else:
        return Ex[xl:xu+1,0,zl:zu+1], Ey[xl:xu+1,0,zl:zu+1], Ez[xl:xu+1,0,zl:zu+1], plot_extent



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