import numpy as np
import h5py as h5
"""
Utilities for parallel Warp operation.
"""


def save_lost_particles(top, comm_world, fsave=None):
    """
    Function for saving lost (scraped) particles coordinate data. Only useful for when Warp is operating in parallel.
    Data is saved in (x,y,z) columns in numpy.save binary file format.
    Due to Warp's dynamic array system data may contain empty coordinates as float 0's.

    Args:
        top: top object from Warp
        comm_world: comm_world object used by Warp
        fsave: path for save file.
            If not given defaults to 'lost_particles_step_$(step_number).npy' in the local directory.

    Returns:

    """
    if not fsave:
        fsave = 'lost_particles_step_{}.npy'.format(top.it)

    xplost_all = []
    yplost_all = []
    zplost_all = []

    # Check that all arrays needed to process locations exist on each rank
    try:
        top.xplost
        top.yplost
        top.zplost
        status = 1
    except:
        status = 0
        "Rank {} does not hold lost particle arrays".format(comm_world.rank)

    # Send each status to head
    all_status = comm_world.gather(status, root=0)

    # Prepare head to receive particles arrays from all valid ranks
    if comm_world.rank == 0:
        for rank, status in enumerate(all_status):
            if status == 1 and rank != 0:
                xplost_all.append(comm_world.recv(source=rank, tag=0))
                yplost_all.append(comm_world.recv(source=rank, tag=1))
                zplost_all.append(comm_world.recv(source=rank, tag=2))

    # All ranks that have valid data send to head
    if status == 1 and comm_world.rank != 0:
        comm_world.send(top.xplost, dest=0, tag=0)
        comm_world.send(top.yplost, dest=0, tag=1)
        comm_world.send(top.zplost, dest=0, tag=2)

    # Process data on head and save
    if comm_world.rank == 0:
        # If the head has particle data then add it to array
        if all_status[0] == 1:
            xplost_all = [top.xplost] + xplost_all
            yplost_all = [top.yplost] + yplost_all
            zplost_all = [top.zplost] + zplost_all
        # If any particle data exists then concatenate all of it together and save
        if len(xplost_all) > 0:
            xplost_all = np.hstack(xplost_all)
            yplost_all = np.hstack(yplost_all)
            zplost_all = np.hstack(zplost_all)
            np.save(fsave, [xplost_all, yplost_all, zplost_all])


def save_pidlost(top, comm_world, species, fsave=None):
    # TODO: Test for if species is iterable
    """
    Function for saving lost (scraped) particles coordinate data. Only useful for when Warp is operating in parallel.
    Data is saved in (x,y,z) columns in numpy.save binary file format.
    Due to Warp's dynamic array system data may contain empty coordinates as float 0's.

    Args:
        top: top object from Warp
        comm_world: comm_world object used by Warp
        fsave: path for save file.
            If not given defaults to 'lost_particles_step_$(step_number).npy' in the local directory.

    Returns:

    """
    if not fsave:
        fsave = 'pidlost_step_{}.h5'.format(top.it)

    particle_data = {}
    for sp in species:
        particle_data[sp.name] = sp

    # Prepare head to receive particles arrays from all valid ranks
    if comm_world.rank == 0:
        received_data = {}
        for sp in particle_data:
            received_data[sp] = []
        for rank in range(1, comm_world.size):
            for sp in particle_data:
                received_data[sp].append(comm_world.recv(source=rank, tag=particle_data[sp].js))

    # All ranks that have valid data send to head
    if comm_world.rank != 0:
        for sp in particle_data:
            comm_world.send(particle_data[sp].pidlost[()], dest=0, tag=particle_data[sp].js)

    # Process data on head and save
    if comm_world.rank == 0:
        for sp in received_data:
            received_data[sp].append(particle_data[sp].pidlost[()])

            # Copy over non-empty arrays
            received_data[sp] = [ar for ar in received_data[sp] if ar.shape[0] != 0]
            if len(received_data[sp]) > 0:
                received_data[sp] = np.row_stack(received_data[sp])
                ff = h5.File(fsave, 'w')
                ff.create_dataset(sp, data=received_data[sp])
                ff.close()
            else:
                print("`save_pidlost`, Step {}: No particles lost".format(top.it))
