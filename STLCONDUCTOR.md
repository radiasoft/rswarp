#### Using the stlconductor module
Currently, Warp's built-in CAD module is not actually implemented yet, but put as a hold. The `stlconductor` module under `rswarp` enables users to define conductor from `.stl` file for Warp.
The `stlconductor` module relies on the [PyMesh](https://github.com/PyMesh/PyMesh) package.

Examples of running Warp with the `stlconductor` can be found at `examples/stlconductor/`. The relevant portions are described below:

```python
from __future__ import division
import numpy as np
from warp import *

from rswarp.stlconductor.stlconductor import *

```

This setup is similar to most Warp simulations, except that we import some utilities from `rswarp`.

```python
install_conductor = True

if install_conductor:
    conductor = STLconductor("tec_grid.stl", voltage=CONDUCTOR_VOLTS, condid=1)

...

if install_conductor :
    installconductor(conductor, dfill=largepos)
```

Here, we first create a `conductor` whose geometry is defined by the file `tec_grid.stl`. Then, the `conductor` is installed in a typical Warp manner.

