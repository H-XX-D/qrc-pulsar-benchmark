# Cislunar cause accounting and measured implementation advantage

The remaining error budget is **not closed**: no available evidence uniquely
determines every maneuver, spacecraft property, reference error or sensor bias.
This report accounts for the major identified causes by separating executed
force ablations, numerical checks, parameter/disturbance scenarios and missing
inputs. Unknown quantities are not fitted to held-out spacecraft states.

The new force pass improves LRO's three-day RMS from 459.78 m to 427.05 m before
adding a lunar-tide convention. The full elastic-tide scenario reaches 367.70 m,
while an epoch-subtracted tide gives 430.53 m. The smaller score alone cannot
choose the correct gravity-field tide convention. Under matched numerical
accuracy, the current spatial shape implementation is slower than Jacobi in
both tested cases. Neither the physics repairs nor the C++ backend establishes
a coordinate-specific advantage.

## Executed physical models

`cislunar_perturbations.py` supplies explicit corrections to the existing lunar
harmonics, Earth/Sun differential gravity and configured SRP. The replay's new
`eval_state_acceleration(r,v,t)` interface includes velocity-dependent forces;
the orbit/clock filter can use that same interface and its velocity Jacobian.
Position-only callers retain their existing `eval_total_acceleration(r,t)`
contract and do not silently receive a velocity-dependent model.

The additions are:

- Earth's J2 with an IAU 2006/2000A celestial pole and the conventional normalized
  C20 value from [IERS Chapter 6](https://iers-conventions.obspm.fr/content/chapter6/icc6.pdf).
  Its acceleration at the lunar origin is subtracted as well as at the spacecraft.
- Mercury, Venus, Mars, Jupiter, Saturn, Uranus, Neptune and Pluto system tides,
  using preloaded DE440s barycentric states and matching system GMs. The GM reader
  handles Fortran `D` exponents and converts km³/s² to m³/s². Its unit/exponent
  regression test prevents a factor-of-thousands error encountered during development.
- Order-reduced first-PN Einstein–Infeld–Hoffmann corrections for all configured
  point masses, including cross potentials and velocity terms. They are computed
  in barycentric coordinates and the lunar-origin correction is subtracted.
  This follows the structure of [JPL's equations of motion](https://spsweb.fltops.jpl.nasa.gov/portaldataops/mpg/MPG_Docs/Source%20Docs/Standish-Chap-8.pdf),
  rather than summing isolated two-body precession formulas.
- An elastic degree-two lunar response to Earth and Sun, using GRGM900C's
  published k2 = 0.024116. For each tide-raising body, the potential is
  `U = k2 GM_body R_moon^5 P2(cos(angle)) / (distance_body^3 radius_spacecraft^3)`.
  An analytic tensor gradient is tested against an independent potential gradient.
  Two explicitly named background conventions are retained: the full induced
  tensor, and subtraction of the tensor at the initial epoch in lunar body axes.
  The latter is a sensitivity experiment, not a verified permanent-tide conversion.
  Frequency-dependent response, phase lag and higher-degree tides remain outside
  this implementation; see NASA's [lunar tidal-response work](https://pgda.gsfc.nasa.gov/products/96).

Every model uses the same first spacecraft state and scores all subsequent
archive samples over three days. No fitting, state resets or residual-derived
forces are used. Planetary ephemerides are preloaded inputs. The starting
spacecraft state is itself reconstructed and may use later orbit-determination
observations, so this is a propagation replay rather than a strictly operational
blind forecast.

RMS position errors in metres:

| Dataset | Harmonics baseline | + Earth J2, other planets and 1PN | + full elastic tide | + epoch-subtracted tide |
|---|---:|---:|---:|---:|
| LRO Jan 2023 | 459.781 | 427.045 | 367.702 | 430.534 |
| CAPSTONE Feb 2023 | 1,910.592 | 1,912.485 | 1,912.463 | 1,912.485 |
| CAPSTONE Nov 2022 | 341,754.081 | 341,755.897 | 341,755.894 | 341,755.897 |
| Orion Nov 2022 | 9,733,713.439 | 9,733,712.643 | 9,733,712.643 | 9,733,712.643 |

Some physically motivated additions worsen RMS because several errors can
cancel. Models were not selected by the smallest held-out score.

## What can and cannot explain the residual

These are trajectory sensitivities, **not independent standard deviations**.
They must not be added in quadrature or presented as a confidence interval.

| Cause | Executed evidence | Status |
|---|---|---|
| Lunar spatial gravity | Earlier J2-to-degree-120 change reduced LRO RMS 22.749 km to about 0.460 km. Degree 120-to-180 changes position by up to 5.71 m. | Implemented; higher-degree/correlated coefficient uncertainty not fully bounded. |
| First-PN dynamics | Changes LRO position by up to 57.88 m, February CAPSTONE by 7.69 m. | Implemented/tested; multipolar relativity and higher PN orders omitted. |
| Earth J2 | Up to 0.0083 m change for LRO; 13.61 m for February CAPSTONE. | Implemented; full Earth tesseral/time-variable gravity omitted. |
| Other planetary systems | Up to 0.000087 m for LRO; 0.543 m for February CAPSTONE. | Implemented at point-system level; asteroids and planetary multipoles omitted. |
| Lunar elastic tide | Full-tensor addition changes LRO by up to 123.33 m from the other-force model; epoch subtraction changes it by 8.43 m. | Implemented as convention-sensitive scenarios; product-compatible background still unverified. |
| Lunar GM uncertainty | Adding the file's quoted marginal GM uncertainty, 14178.1 m³/s², changes LRO by up to 2.47 m. | Executed one-parameter sensitivity, not full gravity covariance. |
| Integration | Previous degree-120 15-to-7.5-second refinement changes predictions by at most 1.83e-5 m. New runs use the converged 30-second regime for LRO and rtol 2e-12. | Tested numerical convergence; no absolute error certificate. |
| Planetary interpolation | Changing the preload grid from 3600 to 900 seconds changes LRO by 1.5e-6 m and February CAPSTONE by 0.00024 m. | Too small to explain the present residual in these tests. |
| Initial velocity uncertainty | Adding 1 mm/s on one axis changes LRO by 205–629 m after three days; February CAPSTONE by 235–350 m. | Material sensitivity; actual initial covariance unavailable. |
| Initial position uncertainty | Adding 100 m on one axis changes LRO by 27.2–56.5 km; February CAPSTONE by 382–451 m. | Illustrative scenarios, not a claim that the starting state is this uncertain. |
| Persistent unmodeled force | A prescribed 1e-9 m/s² along-track acceleration changes LRO by 101.1 m. The same radial/normal amplitudes give 0.585/0.0045 m; February CAPSTONE's three directions give 28.8–41.4 m. | Direction/history matter; these are hypothetical inputs, never fitted residuals. |
| SRP, attitude, mass, self-shadow, thermal recoil and lunar reflected/IR radiation | Historical LRO SRP with the additional forces gives 461.27 m RMS without tides and 401.89 m with the full-tide scenario. Current-epoch optical/mass/attitude/thermal histories are not verified. | Input-limited. For scale, anisotropic thermal power P gives acceleration at most P/(m c), but P, direction and history are not supplied. |
| Maneuvers, leaks, attitude-control events | Independent reports identify maneuvers in the November CAPSTONE and Orion windows; their full vectors remain unavailable. | Those large failures remain in the scores; no invented corrective burn. |
| Reference reconstruction and sampling | Original NASA FDF SPK agrees with the bundled LRO positions within 0.087 m across all 2,161 epochs. A localized reference acceleration mismatch remains near 4,202 seconds. | Reference provenance checked; the mismatch's physical cause is unresolved. |
| Clock, source position/proper motion, phase ambiguity, dispersion, relativistic light propagation and detector noise | Orbit/clock EKF is tested with idealized synthetic pseudorange observations. No raw-flight observation adapter or calibrated mission noise model is available. | Cannot assign flight accuracy or ground-update savings from this experiment. |
| Finite precision, hardware execution and runtime faults | Host binary64 tests, Python/native parity and prior sanitizers pass. | FPGA fixed-point error, synthesis, power, radiation tolerance and device timing unmeasured. |

This is coverage of the major identified causes, not a proof that no additional
effect exists. Exospheric drag, charging/Lorentz forces, outgassing and small
impulses also require environment/spacecraft information to turn them into
mission-specific bounds. A generic acceleration allowance is useful for
sensitivity testing, but it does not identify which of these mechanisms acted.

## The LRO reference feature

The downloaded [NASA FDF Regular Definitive SPK](https://naif.jpl.nasa.gov/pub/naif/pds/data/lro-l-spice-6-v1.0/lrosp_1000/data/spk/lrorg_2022349_2023074_v01.bsp)
contains daily reconstructed trajectories; its January 1–3 comments explicitly
list no maneuvers. The predictions and source states diverge rapidly near
01:10 TDB. Differentiating the source velocities at 0.25 and 0.125-second
half-widths gives a peak mismatch of approximately **0.05016 m/s²** against the
expanded model, with differencing disagreement below 1.89e-6 m/s² over the
examined window. Position/velocity consistency improves under finer sampling,
so the evidence does not establish a literal position discontinuity.

That feature is far larger than the tested smooth SRP/tide/relativistic terms.
An unlisted force event, reconstruction/interpolation behavior or another input
mismatch remain rivals. The orbit-file maneuver statement does not justify
asserting an actual burn, and the derivative diagnostic is not an accelerometer.
It is not injected into a forecast or removed from any reported score. NASA
describes the broader reconstruction process on its [LRO trajectory page](https://pgda.gsfc.nasa.gov/products/65).

## Advantage estimate

`coordinate_benchmark.py` compares conventional Cartesian, Jacobi and the
existing direct spatial shape/scale/orientation ODE. All three use identical
isolated Newtonian Earth–Moon–spacecraft forces and the same initial state.
This isolates the coordinate implementation; it is not a full-force mission
comparison. Accuracy is measured against a separately refined Cartesian
trajectory, with control changes below 8 micrometres. The benchmark sweeps
tolerances, reconstructs the same 1,001 output states, alternates execution
order, and reports three repeats using both wall and process CPU time.

Fastest tested configuration below **one metre maximum numerical position
error**, macOS ARM64, Python fallback kernels:

| Case | Cartesian CPU | Jacobi CPU | Spatial shape CPU | Shape / Jacobi cost |
|---|---:|---:|---:|---:|
| CAPSTONE initial geometry, three days | 3.813 ms | 3.601 ms | 31.599 ms | **8.78×** |
| LRO initial geometry, six hours | 17.488 ms | 17.248 ms | 166.864 ms | **9.67×** |

At a 0.1 m limit, shape/Jacobi cost ratios are 6.60× and 9.67×. These finite
tolerance sweeps are implementation measurements, not a theoretical lower bound
on either formulation. Direct shape evolution is mathematically equivalent
where its chart is valid; the missing forces affect both formulations. The
current results support **no demonstrated shape-specific speed or accuracy
advantage**, with a measured speed of roughly 0.10–0.11× Jacobi at the one-metre
target. A native implementation or different chart could change the result and
would need the same test.

The current spatial shape ODE has 14 binary64 values, including an unused clock;
removing that clock leaves 13. Jacobi uses 12. The shape coordinates alone omit
scale, spatial orientation and their rates. State storage is 112 bytes versus
96 bytes for Jacobi, compared with **118,096 bytes** for packed degree-120 gravity
coefficients shared by either approach. The two-dimensional projection therefore
does not currently establish an onboard memory saving.

The existing C++ gravity backend does have a measured implementation benefit:
**53.7 microseconds per degree-120 force evaluation**, versus 134 ms for the
unoptimized Python reference, about 2,498× faster with maximum sampled force
disagreement 6.71e-16 m/s². This is not a comparison against an optimized native
conventional solver and is available to either coordinate formulation. No FPGA
performance is inferred from it.

For autonomous estimation, three declared synthetic seeds with the expanded
forces give **134–236 m RMS** during a two-hour observation outage after four
hours of measurements, versus about 71.6 km from the same perturbed initial
state with no updates. The idealized four-direction sensor geometry, 300 m
noise and observation schedule are unchanged. This supports estimator software
feasibility under those assumptions; three seeds do not establish reliability
probabilities, real sensor performance, a shape-specific benefit or a percentage
reduction in ground corrections.

## Reproduction and verification

```sh
python -m qrc_pulsar_bench.cislunar_error_budget \
  --dataset lro_jan2023 --physics-dir /path/to/gravity-and-orientation-files \
  --ephemeris /path/to/de440s.bsp --gm-kernel /path/to/gm_de440.tpc \
  --native-library /path/to/liblunar.so --output ./lro-budget --initial-sensitivity
python -m qrc_pulsar_bench.coordinate_benchmark \
  --dataset capstone_feb2023 --ephemeris /path/to/de440s.bsp \
  --output ./coordinates.json --repeats 3
python -m pytest -q
```

Run the budget command for each of the four supported dataset names. All seven
force models are retained without selecting by held-out RMS. Input hashes, all model options, full
states, uncertainty scenarios and benchmark repetitions are saved in the
[portable results](cislunar-cause-budget-results-2026-10-31.json). Supplementary
scripts/arrays are referenced there under the evidence directory. The previous
[refinement guide](cislunar-force-refinement.md) gives public input download and
native compilation commands.

The integrated suite passed **261 tests** and the wheel built with all new
modules. Tests cover the independent tidal gradient, vectorized/loop EIH parity,
the existing analytic relativistic limits, Fortran GM exponents, velocity-dependent
propagation and covariance, future-value invariance and existing native bridges.
No C ABI changed in this pass. Recall's impact walk was continued at its capped
boundary; the broader pulsar/shape tests passed, and the preregistered ARTEMIS
forecast and unrelated mission-planning work were not changed by this analysis.
