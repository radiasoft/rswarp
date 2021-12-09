
# Install mystic, if you don't have it pip install mystic


import numpy as np
from mystic.tools import random_state


def _random_samples(lb,ub,npts=1000):
  """
generate npts random samples between given lb & ub
Inputs:
    lower bounds  --  a list of the lower bounds
    upper bounds  --  a list of the upper bounds
    npts  --  number of sample points [default = 10000]
"""
  dim = len(lb)
  pts = random_state(module='numpy.random').rand(dim,npts)
  for i in range(dim):
    pts[i] = (pts[i] * abs(ub[i] - lb[i])) + lb[i]
    
  # np.savetxt("training.txt", pts.T)
  np.save( 'mesh_file.npy', pts)

  return pts
  
  
# Bounds of the parameter space: sheet_density and beam_charge
lb=[0.5e-9, 40e-6, 20e-6]
ub=[1.5e-9, 60e-6, 35e-6]

data=_random_samples(lb,ub,npts=500)
#
