from copy import deepcopy
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt


def convertunits(particlearray):
    """
    Putting particle coordinate data in good ol'fashioned accelerator units:
        x: m
        x': ux/uz
        y: m
        y': uy/uz
        z: m
        p: MeV/c

    """
    dat = deepcopy(particlearray)  # Don't copy by reference
    dat[:, 1] = dat[:, 1] / dat[:, 5]
    dat[:, 2] = dat[:, 2] / dat[:, 5]
    dat[:, 5] = dat[:, 5] / 5.344286E-22

    return dat


def plotphasespace(particlearray):
    fig = plt.figure(figsize=(12, 8))

    ax0 = plt.subplot(2, 2, 1)
    ax0.scatter(particlearray[:, 0], particlearray[:, 1])
    ax0.set_title("x-xp phase space")
    ax0.set_xlabel("x (m)")
    ax0.set_ylabel("xp (rad)")

    ax1 = plt.subplot(2, 2, 2)
    ax1.scatter(particlearray[:, 2], particlearray[:, 3])
    ax1.set_title("y-yp phase space")
    ax1.set_xlabel("y (m)")
    ax1.set_ylabel("yp (rad)")

    ax2 = plt.subplot(2, 2, 3)
    ax2.scatter(particlearray[:, 0], particlearray[:, 2])
    ax2.set_title("x-y distribution")
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")

    ax3 = plt.subplot(2, 2, 4)
    ax3.scatter(particlearray[:, 4], particlearray[:, 5])
    ax3.set_title("z-pz phase space")
    ax3.set_xlabel("z (m)")
    ax3.set_ylabel("pz (MeV/c)")

    fig.tight_layout()
    plt.show()

