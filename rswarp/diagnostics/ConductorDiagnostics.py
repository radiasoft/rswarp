from scipy.constants import e
import numpy as np


def analyze_scraped_particles(top, particles, solver):
    """
    Returns dictionary with conductor id's as keys. Each entry
    contains an array of form [step, particles deposited on step]
    (Assumes only one species present in simulation)
    Args:
        top: Warp's top object.
        particles: Species object for particles.
        solver: Electrostatic solver object.

    Returns:
        [step, particles deposited on step]
    """
    cond_ids = []
    cond_objs = []
    lost = {}
    for cond in solver.conductordatalist:
        cond_objs.append(cond[0])
        cond_ids.append(cond[0].condid)

    for i, ids in enumerate(cond_ids):
        lost[ids] = np.copy(cond_objs[i].lostparticles_data[:, 0:4])
        lost[ids][:, 0] = np.ndarray.astype(np.round(lost[ids][:, 0] / top.dt), 'int')
        lost[ids][:, 1] = np.round(-1. * lost[ids][:, 1] / particles.sw / e)

    return lost
