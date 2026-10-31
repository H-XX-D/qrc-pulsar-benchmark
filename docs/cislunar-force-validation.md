# Cislunar force repair and forward validation

This page preserves the first force-repair experiment. The subsequent
[gravity, maneuver, SRP and orbit/clock refinement](cislunar-force-refinement.md)
adds a spatial lunar field and a separate synthetic navigation experiment.

The mission replay now propagates explicit physical forces from an initial
spacecraft state. Future spacecraft samples are used only to score a forecast.
The implementation remains an inertial position/velocity solver; these results
establish neither an advantage from direct shape-sphere evolution nor flight
navigation or FPGA qualification.

## Physical model and interfaces

`cislunar_forces.py` uses metres, seconds and Moon-centred nonrotating ICRF axes.
Epochs are JD **TDB**, not UTC. The spacecraft acceleration is:

```
a = -GM_moon r / |r|^3
    + GM_earth [(R_earth-r)/|R_earth-r|^3 - R_earth/|R_earth|^3]
    + GM_sun   [(R_sun-r)/|R_sun-r|^3     - R_sun/|R_sun|^3]
    + a_J2 + a_SRP

a_J2 = (3/2) J2 GM_moon R_moon^2 / |r|^5
       * [(5 (r·k)^2/|r|^2 - 1) r - 2 (r·k) k]
```

The indirect terms subtract the acceleration of the lunar origin. `k` is the
Moon's unit pole in ICRF, evaluated from the IAU orientation series in
[NAIF pck00011.tpc](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/pck00011.tpc).
The analytic pole was independently checked against CSPICE `pxform` at three
epochs, agreeing within 1.2e-16 per component. It is not a binary lunar PCK model.

The lunar GM changed from 4.9048695e12 to **4.9028001184575496e12 m³/s²**;
Earth and Sun use the matching
[DE440 GMs](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/gm_de440.tpc).
The J2 coefficient remains an explicit approximate value, 2e-4, with reference
radius 1737.4 km. The old vector formula had both an incorrect radial power and
an incorrect polar sign. Tests now compare against an independent gradient of
the quadrupole potential, including rotated axes.

Solar gravity and radiation pressure are separate. Cannonball SRP is
`P_1AU * Cr * area/mass * (AU/d)^2`, directed away from the Sun. Moon/Earth
occultation switches it off in full shadow. The subsequent refinement adds
uniform-disc partial and annular eclipses in both Python and the native kernel;
simultaneous Earth/Moon obscuration is approximated by the larger individual one.
**Default area/mass is zero**, so SRP stays disabled until supplied explicitly.
Unknown properties are not estimated from the future reference trajectory.

`PlanetaryEphemeris` preloads Earth/Sun positions and velocities on a fixed
one-hour grid and uses cubic Hermite interpolation. This grid is independent of
spacecraft sampling cadence. The default Astropy `builtin` ephemeris is an
approximate convenience model; the reported mission comparisons used a local
DE440s kernel. Queries outside preload coverage raise an error. DOP853 uses
error control, checks lunar impact and rejects invalid times/steps.

```python
from qrc_pulsar_bench.capstone_replay import MultiDatasetFlightReplay
from qrc_pulsar_bench.cislunar_forces import CislunarForceParameters

replay = MultiDatasetFlightReplay(
    "capstone_feb2023",
    ephemeris="/path/to/de440s.bsp",
    parameters=CislunarForceParameters(),  # SRP disabled until properties supplied
)
result = replay.run_forward(duration_days=3)
print(result["rms_position_error_m"])
```

`run_causal_replay` and `CapstoneFlightReplay.run_replay` remain legacy API names,
but a controller-enabled run is labelled `offline_reference_feedback` and
`causal_integrity_verified=False`. Reference interpolation is ideal offline
truth, not an onboard sensor or estimator. Ground-connection requirements are
unknown (`None`). Legacy “disturbance” summary fields are sampled residual
magnitudes, not identified physical accelerations.

The separate CR3BP station-keeping experiment now uses the correct normalized
J2 expression. Its pole is approximated by rotating-frame +z and its prescribed
SRP is synthetic; it does not include solar gravity. Multi-path rollouts can use
`MultiPathTrajectoryOptimizer(acceleration=replay.eval_total_acceleration)` with
`start_time_s` for the same ephemeris force model. Its default is still Moon-only.
The final rollout endpoint now participates in the altitude check; discrete
checks do not certify the entire continuous trajectory between samples.

## Why a sampled residual was not a measured force

The prior replay differentiated archived velocities, subtracted a limited
Moon/Earth model, then added that difference to every forward force evaluation.
Changing only the first *future* CAPSTONE velocity sample by +1 m/s changed the
first-hour forecast by **1,500.29 m**, despite identical initial conditions.
The repaired force and forecast pass exact future-value-invariance tests, even
when later positions are replaced by grossly different values and access to
reference/diagnostic methods is made to raise exceptions.

A pure circular lunar two-body orbit, with no unmodeled force, produces about
**0.00281 m/s²** of apparent disturbance under the old 120-second central
velocity difference. That is comparable to the old LRO residual magnitude.
The diagnostic now differentiates with a five-point local polynomial using
actual timestamps and reports disagreement with a three-point stencil. The
latter is a resolution indicator, **not an error bound**. No diagnostic residual
enters the propagator. Maneuvers, sampling error, reconstruction discontinuities,
initial-state errors and missing forces remain competing explanations.

## Executed three-day comparisons, 2026-10-31

Every row starts from the same first reconstructed spacecraft state and scores
all subsequent archived states through day three. No resets, fitted epochs,
future-derived accelerations, trajectory fitting, burns or SRP were used.
Planetary ephemerides are preloaded model inputs, so this is not an isolated
four-body initial-value calculation. These are comparisons to reconstructed
Horizons ephemerides, not raw navigation observations.

RMS position error, kilometres:

| Dataset | Previous GM, Moon/Earth only | DE440 GMs, Moon/Earth | + Sun | + Sun and lunar J2 |
|---|---:|---:|---:|---:|
| CAPSTONE Nov 2022 | 337.944 | 337.488 | 341.761 | 341.755 |
| CAPSTONE Feb 2023 | 35.648 | 36.813 | 1.952 | 1.916 |
| Artemis I Orion | 9,694.682 | 9,695.623 | 9,733.714 | 9,733.713 |
| LRO Jan 2023 | 240.758 | 38.091 | 38.112 | 22.749 |

“Previous GM” is a controlled physical-force ablation using the old constants,
not the old leakage-contaminated replay result. All columns share the repaired
TDB ephemerides and numerical integration. LRO improves about 10.6 times and
February CAPSTONE about 18.6 times relative to that baseline. Refining the solver
from 600 to 300 seconds maximum step and tightening rtol from 1e-11 to 2e-12
changes predictions by at most **2.62 mm** across these cases (LRO); the remaining
kilometre errors are not explained by this tested integration refinement.

Large failures remain visible. November CAPSTONE tracks to 234 m at 24 hours,
then diverges strongly; a maneuver or reference discontinuity is a candidate,
not a uniquely established cause. NASA documents orbit-refinement maneuvers
in that arrival week in its
[CAPSTONE mission update](https://www.nasa.gov/news-release/capstone-forges-new-path-for-nasas-future-artemis-moon-missions/).
The Orion window contains a documented November 25 insertion burn, explicitly
absent from the coast model; see
[NASA's flight-day report](https://www.nasa.gov/blogs/missions/2022/11/25/artemis-i-flight-day-10-orion-enters-distant-retrograde-orbit/).
Neither entire window is a clean test of unforced coasting. No samples were
removed or used to estimate a corrective force to hide those failures.

The four initial state vectors were independently re-queried from Horizons with
Moon centre, ICRF axes, TDB and no light-time correction and matched exactly.
This verifies the starting frame/units, not every archive sample or its covariance.
[Provenance and hashes](cislunar-dataset-provenance.json) and
[full numerical summary](cislunar-force-results-2026-10-31.json) are included.
Reproduce with:

```sh
python -m pip install '.[test,navigation]'
python -m qrc_pulsar_bench.cislunar_validation \
  --ephemeris /path/to/de440s.bsp --days 3 --output ./force-results
python -m pytest tests/test_cislunar_forces.py tests/test_capstone_replay.py
```

## C++ contract and hardware evidence

`hardware/cislunar_force_core.h` is a distinct C ABI:

- Contiguous binary64 arrays: position/velocity length 3, environment length 9,
  force parameters length 7. Layout and units are specified in the header.
- Caller-owned buffers; no allocation, callbacks, retained pointers or global
  mutable state. Calls with independent buffers are reentrant.
- Return 0 for success, 1 for invalid numeric inputs/collision; outputs unchanged
  on failure. Raw C callers must supply valid pointers and the stated lengths.
  The Python wrapper checks shapes and translates failure to `ValueError`.
- `cislunar_verlet_step` receives environment values at both step endpoints and
  is **second order**. Its errors converge by approximately 4x per timestep
  halving against an independent, time-dependent DOP853 calculation. It has no
  continuous collision certificate between endpoints.
- The original `jacobi_symplectic_step` ABI is preserved. It evolves three
  barycentric Cartesian point masses with fourth-order Yoshida; it has neither
  external cislunar forces nor an energy watchdog.

Compile the new host library and load it explicitly through
`NativeCislunarKernel`. The bridge is tested against Python forces over rotated
states, SRP/eclipse cases, invalid inputs and step convergence. The two kernels
serve different coordinate/state contracts and are not interchangeable.

```sh
clang++ -std=c++17 -O2 -shared -fPIC \
  hardware/cislunar_force_core.cpp -o libcislunar.so
```

`run_cislunar_hls.tcl` supplies a synthesis entry point. No vendor HLS tool,
RTL co-simulation, routed timing, resource report, power measurement or device
execution was available in this run. The older `FPGAOrbitEngine` cycle/resource
figures are assumptions; its error against a reference trajectory is now marked
unknown instead of a literal zero. Passing host tests does not verify FPGA speed.

## Remaining force and estimation limits

The first experiment used J2 only. The subsequent refinement implements a
normalized GRAIL harmonic field and matching lunar orientation, explicit finite
burns, plate SRP, partial shadow and an orbit/clock estimator. Independent
epoch-specific maneuver and spacecraft properties are still missing for the
reported arcs, and the estimator has only a synthetic measurement demonstration.
Thermal recoil, Earth oblateness, additional planetary perturbations, solid-body
tides and relativity remain outside this force model. Initial-state covariance
and actual onboard sensor validation are still needed.

The dependency review also reached interbody transfers, Jovian tours, sample
return and GEO capture. Their regression tests pass, but literal targeting
errors, assumed FPGA latencies, heuristic transfer budgets and mission outcome
strings are not validated by these cislunar repairs. No flight-certified claim
is inferred from passing those tests.
