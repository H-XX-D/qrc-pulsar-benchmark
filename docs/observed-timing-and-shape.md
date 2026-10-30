# Observed timing and direct shape evolution

These are experimental, explicit alternatives to the original illustrative
`simulate` command. They preserve the established Newtonian dynamics and add
separately selectable observation and first post-Newtonian models. They do not
turn previously published residuals into predictions.

## Prepare and fit real arrivals

```sh
python -m pip install -e '.[test,timing,acceleration]'
python examples/prepare_observed_arrivals.py --tim /path/to/publisher.tim --output prepared
python examples/fit_observed_arrivals.py --input prepared/timing-inputs.npz \
  --output baseline --cutoff-mjd 56436.5
python examples/fit_observed_arrivals.py --input prepared/timing-inputs.npz \
  --output expanded --cutoff-mjd 57052.45 --fit-orbit all \
  --astrometry motion --astrometry-priors --dm-order 1 --post-newtonian
```

The preparation example is specifically for the full-precision J0337 release
at <https://zenodo.org/records/1241122>. It verifies the original file checksum,
retains only measured timing fields and pulse-number labels, and recomputes
barycentric times and astrometric derivatives with PINT. It never reads the
release's residuals or model-derivative flags as fit inputs. The reference time
is MJD 55920 TCB and the starting elements are Ransom et al. (2014), Table 1.
PINT's intermediate TDB times are explicitly converted to TCB. Do not combine
these numerical orbital parameters with unconverted TDB epochs.

The fitter estimates up to 14 orbital parameters jointly, using scaled finite
differences and variable projection of phase, spin, two telescope offsets,
dispersion and optional astrometry. Parameters are fitted only on the specified
early arrays. `evaluate_frozen` evaluates later arrays without changing them.
`--fit-orbit epochs` adjusts only the two starting epochs. Linear and quadratic
DM trends are options, not a guarantee that future plasma variations are
predictable. Proper-motion columns describe Solar-System astrometric delays;
the orbital viewing direction is currently fixed (no Kopeikin reconstruction).
The astrometric approximation is linearized around the stated position/epoch.

`--astrometry-priors` adds the early VLBA position errors and the approximate
photometric-distance constraint from Ransom et al. (2014). The optical distance
is model dependent, including the companion mass. Published marginal orbital
errors are optimization scales only, not independent Gaussian priors: the
orbital parameters are correlated. Unconstrained early fits can extrapolate
very poorly even when their calibration residuals improve. Compare methods
using the same cutoff and report both ordinary and weighted RMS. Pulse counts
are supplied phase connections; they are not independently recovered here.

`relativity.eih_correction` implements the full three-mass potential and
velocity terms through first PN order (GR beta=gamma=1), with Newtonian
accelerations substituted inside the PN correction. It is not an isolated
binary precession approximation. The timing trajectory retains the
mass-weighted center-of-mass position/velocity, which need not stay at rest
under these equations. The clock integral includes the full pulsar velocity.
It does not include 2PN, radiation reaction, stellar tides, a full Galactic
timing model or arbitrary stochastic dispersion evolution.

## Evolve shape directly

```python
from qrc_pulsar_bench.observed_timing import initial_state, observables
from qrc_pulsar_bench.shape_dynamics import integrate_spatial_shape

state, metadata = initial_state()
trajectory = integrate_spatial_shape(state, end_seconds=86400, rtol=1e-12)
shape = trajectory.shape([0, 43200, 86400])
physical_state = trajectory.sol([0, 43200, 86400])
```

The new ODE states contain shape coordinates, a scale coordinate, orientation,
and their rates. Reconstructed Jacobi vectors are outputs and algebraic force
inputs, not the integrated state. Shape alone is insufficient for a physical
orbit or arrival delay.

`integrate_planar_shape` evolves a mass-weighted stereographic coordinate
`u=q2/q1`, log scale and rotation, and their rates. It retains nonzero angular
momentum and signed orientation, and passes through collinear triangles. It
rejects nonplanar initial positions/velocities. Binary collision `q1=0` is
outside its chart; there is no collision regularization.

`integrate_spatial_shape` retains a quaternion orientation and body-frame
angular velocity, so noncoplanar motion is supported. Its triangle frame
requires a noncollinear triangle. Close approaches to collinearity make this
chart ill conditioned, and its cost/error can be worse than Jacobi integration.
The spatial shape quotient uses a hemisphere because a reflected triangle is
related by a spatial rotation. This is distinct from the oriented planar S².
No automatic fallback hides a Cartesian/Jacobi integration inside this path.
Neither direct shape method currently includes first-PN forces.

The source derives the coordinate equations explicitly. Tests compare full
physical trajectories and velocities, spatial rotation, oriented equator
crossings, Newtonian force parity, analytic relativistic limits, and synthetic
orbital-parameter recovery. The sphere identity alone is not a dynamics test.
Jacobi remains the default until an advantage is demonstrated at matched
physical error, on a specified problem. Optional Numba acceleration changes
execution speed, not the equations; the pure-Python fallback remains available.
