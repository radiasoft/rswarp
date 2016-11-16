#!/usr/bin/env python
"""
Utilities for beam transformation
"""
from __future__ import division
import numpy as np


def rotationMatrix3D(rotaxis, theta):
    """
    Generates a 3D rotation of angle $\theta$ about the specified axis

    rotaxis - vector (x, y, z) specifying the axis of rotation
    theta - angle (in radians) of rotation
    """
    # print("In rotationMarix3D(): rotaxis={}, theta={}".format(rotaxis, theta))
    assert np.ndim(rotaxis) == np.ndim(theta)+1 == 1, "Dimension mismatch in rotationMatrix3D()"
    norm = np.linalg.norm(rotaxis)
    assert norm > 0, "Rotation axis has norm of zero"
    rotaxis = np.array(rotaxis) / norm
    ux = rotaxis[0]
    uy = rotaxis[1]
    uz = rotaxis[2]
    costh = np.cos(theta)
    sinth = np.sin(theta)

    R = np.matrix('{} {} {}; {} {} {}; {} {} {}'.format(
        costh+ux**2 * (1 - costh), ux*uy*(1-costh) - uz*sinth, ux*uz*(1-costh) + uy*sinth,
        uy*ux*(1-costh) + uz*sinth, costh + uy**2 * (1-costh), uy*uz*(1-costh) - ux*sinth,
        uz*ux*(1-costh) - uy*sinth, uz*uy * (1-costh) + ux*sinth, costh + uz**2 * (1-costh)
    ))

    return R


def rotateVec(vec, rotaxis, theta):
    """
    Given a 3-vector vec, rotate about rotaxis by $\theta$

    Also accepts iterable input if all arguments are compatible lengths
    """
    if np.shape(vec) == (3, ):
        R = rotationMatrix3D(rotaxis, theta)
        norm = np.linalg.norm(vec)
        res = np.dot(R, vec)
        assert np.isclose(np.linalg.norm(res), norm), "Rotation changed vector norm"
        return np.dot(R, vec)
    else:
        assert np.shape(vec)[0] == np.shape(rotaxis)[0] == np.shape(theta)[0], "Dimension mismatch in rotateVec()"
        # Unfortunately, seems that np.dot can't be coerced into doing this operation all at once
        # Tried to build a tensor of rotation matrices and use np.einsum, but couldn't get good reuslts.
        # If this becomes slow at any point, it's a good target for optimization.
        res = np.zeros(shape=(np.shape(vec)[0], 3))
        for i, (v, r, t) in enumerate(zip(vec, rotaxis, theta)):
            # print("In rotateVec(): r={}, t={}".format(r, t))
            norm = np.linalg.norm(v)
            R = rotationMatrix3D(r, t)
            res[i] = np.dot(R, v)
            assert np.isclose(np.linalg.norm(res[i]), norm), "Rotation changed vector norm"
        return np.hsplit(res, 3)


def vector_rotation_test():
    assert np.allclose(rotateVec((1, 0, 0), (0, 0, 1), np.pi/2.), (0, 1, 0))
    assert np.allclose(rotateVec((1, 1, 0), (0, 0, 1), np.pi/4.), (0, np.sqrt(2), 0))


def multiple_vector_test():
    vecs = [(1, 0, 0),
            (0, 1, 0),
            (0, 0, 1)]
    rotaxes = [(0, 0, 1),
               (0, 0, 1),
               (0, 0, 1)]
    thetas = [np.pi/4,
              np.pi/4,
              np.pi/4]
    expected = np.hsplit(np.array([(np.sqrt(2)/2., np.sqrt(2)/2., 0),
                                   (-np.sqrt(2)/2., np.sqrt(2)/2., 0),
                                   (0, 0, 1)]), 3)
    res = rotateVec(vecs, rotaxes, thetas)
    assert np.allclose(res, expected)
