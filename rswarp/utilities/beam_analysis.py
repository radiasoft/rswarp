from copy import deepcopy
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import e, c, electron_mass


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
    dat[:, 3] = dat[:, 3] / dat[:, 5]
    dat[:, 5] = dat[:, 5] / 5.344286E-22

    return dat


def get_zcurrent(particle_array, momenta, mesh, particle_weight, dz):
    """
    Find z-directed current on a per cell basis

    particle_array: z positions at a given step
    momenta: particle momenta at a given step in SI units
    mesh: Array of Mesh spacings
    particle_weight: Weight from Warp
    dz: Cell Size
    """

    current = np.zeros_like(mesh)
    velocity = c * momenta / np.sqrt(momenta**2 + (electron_mass * c)**2)

    for index, zval in enumerate(particle_array):
        bucket = np.round(zval/dz)  # value of the bucket/index in the current array
        current[int(bucket)] += velocity[index]

    return current * e * particle_weight / dz
