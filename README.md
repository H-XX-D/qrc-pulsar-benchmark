# qrc-pulsar-benchmark

An MIT-licensed Python toolkit for **post-Keplerian formula evaluation, hierarchical
tidal scale estimates, Newtonian three-body evolution, triangle geometry, and post-fit pulsar residual inspection**.
Python 3.9 or later is required.

## Scope

The binary calculators implement leading-order GR relations using supplied masses
and orbital elements. The triple calculator estimates perturbation scales for
near-circular, tightly hierarchical systems. The geometric shape map describes
unweighted triangles independently of those calculations.

The separate dynamics module evolves Newtonian point-mass trajectories and computes
geometric Roemer delays. It does not implement relativistic three-body forces or a
complete pulse arrival-time model. It has no certified global truncation error,
nonlinear timing fit, correlated noise model, or held-out forecasting evaluation. Published post-fit residuals
describe the upstream researchers' fits; their scatter does not validate this
package's formula estimates. Frozen literature parameters can themselves depend
on the fitted theory and comparison observables.

## Install and run

```sh
git clone https://github.com/H-XX-D/qrc-pulsar-benchmark.git
cd qrc-pulsar-benchmark
python -m pip install -e '.[test]'

qrc-bench list
qrc-bench all
qrc-bench run --system PSR_J0337+1715
qrc-bench simulate --orbits 5 --points 250
qrc-bench compare --orbits 5 --points 250 --tolerance 1e-10
qrc-bench formula-cost --repeats 1000
```

The standalone equivalent is `python run_benchmarks.py all`. The `formula-cost`
command measures this package's scalar formula evaluations on the current
machine, including input checks and result construction. It does not execute
TEMPO, TEMPO2, PINT, or a trajectory integrator, and reports no comparative speedup.

## Time-dependent mechanics and input provenance

`simulate` integrates the 12 Cartesian components of the Jacobi relative state
`(r1, v1, r2, v2)` with SciPy's eighth-order DOP853 method. `r1 = x2 - x1` and
`r2 = x3 - (m1*x1 + m2*x2)/(m1+m2)`. The accelerations are the Jacobi transform
of **ordinary Newtonian inverse-square gravity**. Center-of-mass translation is
removed; orientation, size, velocities, and chaotic relative dynamics remain.
The shape sphere is calculated from the solution, not used as a two-variable ODE.

The time evolution reads only the supplied masses, periods, eccentricities,
inclinations, and optional orbital angles. It does not read residual files,
observed SEP bounds, pulsar spin frequency, or a precomputed trajectory.
Catalog masses/elements are observational inputs, so this is not a parameter-free
prediction. Their presence is explicit in `result['initial_conditions']`.

**The default trajectory is illustrative, not a reconstruction of PSR J0337's
observed orbit.** Both orbits start at zero true anomaly, zero argument of
periapsis, and zero ascending node at an arbitrary `t=0`. No observed epoch is
supplied. Python callers can set `inner_angles_rad` and `outer_angles_rad`, each
`(ascending_node, argument_of_periapsis, true_anomaly)` in radians, and a
`line_of_sight` vector. Orbital inclination rotates the state once; the default
observer is fixed along +z, with geometric delay `-x1_z/c`.

```python
from qrc_pulsar_bench import PSR_J0337_1715, RelationalThreeBodyIntegrator

trajectory = RelationalThreeBodyIntegrator.integrate(
    PSR_J0337_1715, num_inner_orbits=5, num_output_points=250,
    inner_angles_rad=(0.0, 0.1, 0.5), outer_angles_rad=(0.0, 0.2, 1.0),
)
# t_eval_s: seconds from arbitrary zero; sol_y: 12 x N Jacobi state in SI units.
# cartesian_y: reconstructed 18 x N Cartesian state; r1_m/r2_m: full 3 x N vectors.
```

`compare` measures actual Jacobi and 18-component Cartesian integrations from the
same initial conditions. It reports full inner/outer vector differences, all-body
position and velocity differences, Roemer-delay differences, and energy errors
for both solutions. A rotated trajectory with unchanged separations is detected.
Both implementations use DOP853, so agreement is coordinate consistency, not
independent physical validation or proof of equivalent errors at equal tolerances.
Wall times include integration only and are single local samples, not a speedup
claim. Vary tolerances and compare physical errors before drawing performance
conclusions; SI position and velocity components have different units.

COM drift is measured from reconstructed positions (including roundoff), not
hardcoded to zero. Shape coordinates are unsigned, unweighted geometry with
`w3 >= 0`; they fold mirror orientations together and use the same body-label
convention as `ShapeSphereReduction`. The unit-sphere identity and COM constraint
are geometric/construction properties, not evidence of accurate time evolution.

Tests include independent pairwise-force checks, the analytic circular two-body
limit, full-vector convergence, energy and angular momentum, face-on/edge-on
projection, metadata perturbation, and injected phase/rotation and solver failures.
Failed, incomplete, or nonfinite solutions raise instead of appearing successful.
Collision singularities are rejected when encountered; close encounters have no
regularization or guaranteed accuracy. Diagnostic extrema are sampled at output
times, not certified over every internal integration step. `num_steps` remains a
legacy alias for `nfev` (force evaluations, not accepted integration steps).

## Residual inspection

```sh
# Downloads the published CSV only if the destination is missing.
qrc-bench zenodo --path psr_j0337_zenodo_residuals.csv --windows 10

# Optional linearized sensitivity calculation using all supplied derivatives.
qrc-bench zenodo --path psr_j0337_zenodo_residuals.csv --sep-diagnostic

# Arecibo subset; --simulate optionally integrates illustrative orbits over its duration.
qrc-bench arecibo --path psr_j0337_zenodo_residuals.csv

# MJD, residual, uncertainty, followed by optional extra columns.
qrc-bench custom --path my_residuals.csv --format standard --unit us

# MJD, frequency(MHz), residual, uncertainty, followed by optional extra columns.
qrc-bench custom --path data/nanograv/J1614-2230_NANOGrav_12yv4.all.res --format nanograv
```

CSV and whitespace-delimited numeric tables are supported. Comment lines begin
with `#`; an uncommented text header or raw `.tim`/TOA file is not supported.
`--unit` accepts `us`, `ns`, or `s` for both residuals and uncertainties.

The default `--format auto` accepts exactly three columns or the explicit
NANOGrav comment header `MJD frequency(MHz) residual(us) uncertainty(us)`.
Other multi-column tables require an explicit format. Numerical values are never
used to guess which column represents frequency. An automatically detected `us`
header rejects a conflicting unit option.

Each report separates:

- **RMS about zero:** `sqrt(mean(residual**2))`.
- **Standard deviation about the mean:** `std(residual, ddof=0)`.
- Mean residual and an unweighted, whole-dataset linear slope.
- **Chi-square per observation:** `sum((residual/uncertainty)**2)/N`. This is not
  reduced chi-square: the upstream fit's degrees of freedom and noise covariance
  are unknown.
- Chronological windows. Empty windows contain `null` statistics (`n/a` in the
  CLI), and every observation belongs to exactly one window.

The program validates finite data, positive uncertainties, a positive integer
window count, and at least two distinct observation times. A small slope is not a
phase-stability or forecasting verdict; it can miss substantial nonlinear trends.

The [Archibald data release](https://zenodo.org/records/1241122) explicitly identifies
the CSV as post-fit residuals and derivatives, with arrival times insufficiently
precise for fitting. New downloads are checked against the publisher's MD5
`f1f3094c8ce8245eff5cd2fec6e22f5d` before being installed. Existing files are never
replaced based on their size; interrupted or corrupt downloads leave no partial
destination file.

`arecibo` identifies the release's reference backend through the named
`j_WSRT_1400` and `j_GBT_1400` indicators, checked against `-tel AO` in the
publisher's `.tim` file. It inspects post-fit residuals. Optional `--simulate`
uses their duration only: it neither aligns simulated orbital phases to the
observed epoch nor compares predictions with arrivals. Running a simulation
beside residual statistics does not establish a timing fit. The former `nancay`
command mislabeled these 5,355 Arecibo observations. It now raises an explanatory
error; this release contains Arecibo (AO), GBT, and WSRT observations, no Nancay
observations. See [Table 1 and observing description](https://arxiv.org/html/1807.02059v1)
and the checked [backend provenance](data/archibald_backend_provenance.json).

Regression results for the published files:

| Dataset | Rows | Standard deviation (us) | Linear slope (us/year) |
| --- | ---: | ---: | ---: |
| J0337 Archibald CSV | 27,110 | 1.542480 | +0.013500 |
| J1614 NANOGrav committed file | 12,775 | 2.992034 | +0.026471 |
| J0740 NANOGrav committed file | 3,328 | 3.827042 | +0.042083 |

For J0337 the true RMS is **1.542602 us**, slightly larger than its standard
deviation because the mean is nonzero.

### SEP diagnostic and published result

`--sep-diagnostic` uses the named `delta` derivative column and projects out all
other supplied parameter derivatives using a whitened, column-normalized SVD
least-squares calculation. It reports conditional and marginal formal uncertainties
and an incremental correction about the original reference fit. Rank-degenerate
parameters are reported as unidentifiable, with no finite marginal uncertainty.

For the published J0337 CSV, the conditional sigma is approximately `7.30e-13`;
the linearized marginal sigma is approximately `1.91e-7`, after projecting out
39 nuisance columns. These assume independent errors with the supplied standard
deviations. They omit correlated noise, systematic errors, nonlinear refitting,
and the original fit's absolute reference value. **Neither is a new absolute SEP
measurement or evidence establishing Delta = 0.**

The separately labeled literature metadata comes from
[Archibald et al., arXiv v1](https://arxiv.org/abs/1807.02059v1):
`Delta = (-1.09 +/- 0.74)e-6` (68% uncertainty) and `|Delta| < 2.6e-6` (95%).
These values are not inferred by this package.

## Formula calculations and reference provenance

The binary calculations use the usual leading-order periastron advance,
Einstein delay, Shapiro range, geodetic precession, and Peters-Mathews orbital
decay formulas. Stellar masses are in solar units, periods are in days, and
output field names identify time/angle units. `sin_i` is a supplied input.

The conversion uses the [IAU 2015 nominal solar mass parameter](https://arxiv.org/abs/1605.09788),
`GM_sun = 1.3271244e20 m^3/s^2` and `T_sun = GM_sun/c^3`, rather than multiplying
separately rounded values of G and solar mass. This is an exact nominal conversion,
not a claim that the physical solar mass is known exactly.

| Reference system | Provenance and comparison caveats |
| --- | --- |
| J0737-3039 | [Kramer 2021](https://arxiv.org/abs/2112.06795), Table IV and Eq. 42. Observed decay `-1.247920e-12` and intrinsic decay `-1.247752e-12` are kept separate. Leading-order estimates omit corrections discussed in the paper. |
| B1913+16 | [Weisberg & Huang 2016](https://arxiv.org/abs/1606.02744), rounded masses and Table 2 elements, intrinsic decay `-2.398e-12`. The gamma target is omitted because arXiv v1 Table 2 has an ambiguous unit/value pairing. |
| J1738+0333 | [Freire 2012](https://arxiv.org/abs/1205.1450): observed decay `-1.70e-14`, intrinsic decay `-2.59e-14`. The intrinsic calculation excludes kinematic contributions to the observed rate. |
| J0337+1715 | Illustrative hierarchical input snapshot retained from the original package; references are Archibald 2018 and Voisin 2020. These rounded inputs are not a reconstruction of either full fitted ephemeris. |
| Kepler-16b | Binary/planet example associated with Doyle 2011. The inherited `44.1 yr` precession and `1.52 hr` TTV values are labeled supplied comparison metadata; the TTV amplitude is not calculated or independently validated here. |

Reported relative differences are not statistical significance or independent
tests of gravity. Uncertainty propagation and parameter covariance would be
required before drawing those conclusions. The package does not treat copied
inputs or literature values as successful predictions.

The triple tidal amplitudes assume small eccentricities and `a1/a2 << 1`; they
omit angular coefficients, phases, mutual-inclination evolution, and higher-order
couplings. The inner binary GW decay includes its Peters eccentricity factor.
`baseline_years` controls a separate cumulative GW delay **scale**, not an actual
time series. Non-separated inner and outer orbits are rejected, but separation
alone is not a certification of the approximation's accuracy. The octupole scale
is signed under relabeling the inner bodies.

## Shape geometry

For side lengths `a=r12`, `b=r23`, `c=r31`, let `S=a*a+b*b+c*c` and `A` be
the triangle's area. The equal-mass/unweighted shape coordinates are

```
w1 = (2*c*c - a*a - b*b) / S
w2 = sqrt(3) * (a*a - b*b) / S
w3 = orientation * 4*sqrt(3)*A / S
```

They agree with normalized Jacobi-vector geometry; collinear shapes have `w3=0`.
Side lengths do not determine oriented planar shape, so the default selects the
upper hemisphere; `orientation=-1` supplies the reflected choice. Scale is
normalized before the area calculation, and invalid distances are rejected.
The map does not retain physical size, velocity, or unequal-mass dynamics.
See [Montgomery's shape-sphere exposition](https://arxiv.org/abs/1402.0841).

Legendre orthogonality is tested only as a basis identity. It does not imply
nonlinear decoupling: `integral(P2*P4)=0`, while `integral(P2**2*P4)=4/35` over
`[-1,1]`. A dynamical claim needs the actual evolution operator and its error tests.

## Python API and compatibility notes

```python
from qrc_pulsar_bench.engine import RelationalTripleEngine
from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.residuals import ZenodoResidualEvaluator

scales = RelationalTripleEngine.solve(PSR_J0337_1715, baseline_years=10)
report = ZenodoResidualEvaluator.evaluate(
    "psr_j0337_zenodo_residuals.csv", include_sep_diagnostic=True
)
```

`ZenodoWalkForwardEvaluator` remains an alias for `ZenodoResidualEvaluator`,
including its old import path in `engine.py`. It does not perform forecasting.
The repair changes these result fields intentionally:

| Previous field | Replacement / meaning |
| --- | --- |
| `overall_rms_us`, `rms_us` | Now true RMS; use `overall_std_us`, `std_us` for the former statistic |
| `global_chi2_dof`, window `chi2_dof` | `chi2_per_observation`, without assumed degrees of freedom |
| Window `mean_drift_us` | `mean_us`, a residual mean rather than a drift rate |
| `delta_obs`, `sigma_delta`, `pull_qrc` | Removed absolute-SEP inference; optional `sep_diagnostic` gives conditional/marginal sensitivity and an increment |
| `delta_sep_qrc` / flat published SEP fields | `published_sep` metadata with explicit source and confidence levels |
| `s_pred` | `s_input` |
| `ttv_semi_amplitude_hours` | `ttv_reference_hours` |
| `delta_t_GW_5yr_ns` | `gw_delay_scale_ns` and `baseline_years` |

The old `agreements_pct`/`agreement_pct` values remain for compatibility only.
Prefer `relative_difference_pct`; it is `null` when the reference is zero.
Ambiguous multi-column data now requires an explicit schema. Supplying a missing
Zenodo path to `RelationalTimingEngine.run_all` now raises instead of silently
skipping the requested analysis.

## Tests

```sh
python -m pytest -ra

# Require the external dataset; missing data is a failure in this mode.
QRC_REQUIRE_ZENODO=1 QRC_ZENODO_CSV=/path/to/psr_j0337_zenodo_residuals.csv \
    python -m pytest -ra
```

The two small NANOGrav files are committed and always tested. The optional Zenodo
test explicitly skips when the CSV is unavailable; an explicitly configured
missing path fails. Tests cover formula symmetries and limits, independent
Jacobi geometry, correlated-nuisance signal recovery, statistics and file schemas,
Newtonian force laws and full-vector dynamics, all CLI paths, and download
integrity/failure behavior. CI is configured for
Python 3.9 and 3.13; a configuration is not evidence that remote CI has run.

## Development and acknowledgments

Core modules are `engine.py` (formula estimates), `dynamics.py` (Newtonian
integration and coordinate comparison), `residuals.py` (inspection and
linear diagnostics), `shape_sphere.py` (geometry), `datasets.py` (download),
`systems.py` (example metadata), and `cli.py`. Tests are under `tests/`.

Further scientific validation needs independently implemented integrators at
matched physical accuracy, long-term/close-encounter error studies, relativistic
forces and timing effects, fully sourced initial conditions at a stated epoch, and
held-out arrival times with a timing/noise model.

Thanks to the cited researchers, observing teams, and the NANOGrav collaboration
for the equations, observations, and public datasets. License: [MIT](LICENSE).
