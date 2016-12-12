#### Using the Ionization module
Currently, the primary feature of `rswarp` is the extended `Ionization` class. Warp's built-in `Ionization` class allows interaction and creation of arbitrary species in an impact ionization event, but is missing some features, which this extended class aims to implement.

An example of a Warp simulation with ionization can be found at `examples/ionization.py`. The relevant portions are described below:

```python
from __future__ import division
import numpy as np
from warp import *
from rswarp.utilities.ionization import Ionization
from rswarp.utilities.beam_distributions import createKV

simulateIonization = True

beam_ke = 100  # beam kinetic energy, in eV
beam_gamma = beam_ke/511e3 + 1
beam_beta = np.sqrt(1-1/beam_gamma**2)
sw = 1
```

This is setup similar to most Warp simulations, except that we import some utilities from `rswarp`.

```python
beam = Species(type=Electron, name='e-', weight=0)
# These two species represent the emitted particles
h2plus = Species(type=Dihydrogen, charge_state=+1, name='H2+', weight=0)
emittedelec = Species(type=Electron, name='emitted e-', weight=0)
```

Here, we create the species that will participate in the ionization event. For this example, we treat the incident and emitted electrons as separate `Species` objects, but it is possible to specify the same `Species` object for both, so that cascading events may be simulated.

```python
...

if simulateIonization is True:
    target_pressure = 1  # in Pa
    target_temp = 273  # in K
    target_density = target_pressure / boltzmann / target_temp  # in 1/m^3

    ioniz = Ionization(
        stride=100,
        xmin=w3d.xmmin,
        xmax=w3d.xmmax,
        ymin=w3d.ymmin,
        ymax=w3d.ymmax,
        zmin=(w3d.zmmin + w3d.zmmax)/2. - w3d.dz*3,
        zmax=(w3d.zmmin + w3d.zmmax)/2. + w3d.dz*3,
        nx=w3d.nx,
        ny=w3d.ny,
        nz=w3d.nz,
        l_verbose=True
    )
```

The initialization routine of the `Ionization` object is identical to Warp. We also calculate the number density of the target gas in terms of more useful engineering parameters here.

```python
    ioniz.add(
        incident_species=beam,
        emitted_species=[h2plus, emittedelec],
```

If we replaced `emittedelec` with `beam` here, we would allow for an arbitrary number of additional ionization events.

```python
        cross_section=h2crosssections.h2_ioniz_crosssection,
        # cross_section=lambda nnew, vi: 1e-20,
        emitted_energy0=[0, h2crosssections.ejectedEnergy],
        # emitted_energy0=[0, lambda nnew, vi: 1./np.sqrt(1.-((vi/2.)/clight)**2) * emass*clight/jperev],
        emitted_energy_sigma=[0, 0],
        # sampleEmittedAngle=lambda nnew, emitted_energy, incident_energy: np.random.uniform(0, 2*np.pi, size=nnew),
        sampleEmittedAngle=h2crosssections.generateAngle,
        # sampleIncidentAngle=lambda nnew, emitted_energy, incident_energy, emitted_theta: np.random.uniform(0, 2*np.pi, size=nnew),
```

Here, specify the influence of the total, singly-differential, and doubly-differential cross-sections.  Warp's built-in behavior allows specifying `cross_section` as either a constant, or a callable function with a single argument `vi`, an `ndarray` containing the velocities of the incident particles. In `rswarp`, this ability to pass a callable (or list thereof) also applies to `emitted_energy0` and `emitted_energy_sigma`, with the arguments as shown in the commented-out lambdas above. The example as written uses the [rsoopic](https://github.com/radiasoft/rsoopic) module for its functions describing these distributions for hydrogen.

`sampleIncidentAngle` and `sampleEmittedAngle` allow the user to specify a callable for specifying the angles (relative to the incident particle's trajectory) of the incident and emitted particles after ionization, respectively. The arguments to these functions are the number of particles `nnew`, and the emitted and incident energies. `sampleIncidentAngle` also takes an additional argument for the angles given to the emitted particles after calling `sampleEmittedAngle` (this means `sampleIncidentAngle` will only be called if `sampleEmittedAngle` is specified). Together, these two functions allow the user to implement emission behavior that resolves momentum transferred to an emitted particle.

Breaking from Warp's behavior, if these sample functions are *not* specified, the emitted particle will be traveling in the *same direction* as the incident particle, with a velocity appropriate to its selected energy.

```python
        writeAngleDataDir=diagDir + '/angleDiagnostic',
        writeAnglePeriod=10,
```

These parameters control writing out of angles of emitted particles. These diagnostics are moderately *slow* at present, so it is best to leave them disabled unless benchmarking the class's behavior.

```python
        l_remove_incident=False,
        l_remove_target=False,
        ndens=target_density
    )
```

The `Ionization.generate()` method is the focus of these extensions, and attention has been given to making the necessary changes in a way that allows reuse for other ionization problems.  Specifically, the portion of this method that generates new particles when an ionization event occurs has been extracted into the `Ionization.generateEmittedVelocity` method, and the method modified to check if the `emitted_energy0` and `emitted_energy_sigma` arguments of passed to `Ionization.add` are callable, and if so, uses them to generate emission energy on-demand in the same way that a callable can be passed for the `cross_section` argument.  In this way, any user of this extended `Ionization` class can specify arbitrary emission energy distributions, likely forgoing setting `emitted_energy_sigma` at all in favor of handling the spread of the desired input in `emitted_energy0` itself.
