import numpy as np
"""
Utilities for parallel Warp operation.
"""

def save_lost_particles(top, comm_world, fsave=None):
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
        "{} failed try".format(comm_world.rank)

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
