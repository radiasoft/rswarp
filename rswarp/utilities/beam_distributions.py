import numpy as np
from random import random


def createKV(npart, a, b, emitx, emity):
    """

    Args:
        npart: Number of particles in distribution.
        a: Horizontal beam size (full, not rms).
        b: Vertical beam size (full, not rms).
        emitx: Full, geometric emittance for horizontal.
        emity: Full, geometric emittance for vertical.

    Returns:
        Array of size [npart,4] for transverse particle coordinates in
        x,x',y,y'.
    """
    ptcls = []


    # twiss alpha - Currently must be zero to give correct distr.
    alphax = 0
    alphay = 0
    betax = a**2 / emitx
    betay = b**2 / emity
    # twiss gamma - For eventual full generalization
    gammax = (1 + alphax**2) / betax
    gammay = (1 + alphay**2) / betay

    i = 0
    while i < npart:
        x = y = 20 * max([a, b])
        while (x / a)**2 + (y / b)**2 > 1:
            x = (1.0 - 2.0 * random()) * a
            y = (1.0 - 2.0 * random()) * b

        R = 1 - (x / a)**2 - (y / b)**2

        theta = random() * 2 * np.pi

        xp = np.cos(theta) * (np.sqrt(R) * emitx) / a
        yp = np.sin(theta) * (np.sqrt(R) * emity) / b

        ptcls.append([x, xp, y, yp])
        i += 1
        # print (x / a)**2 + (y / b)**2 + (xp * a / emit)**2 + (yp * b / emit)**2

    return np.array(ptcls)
