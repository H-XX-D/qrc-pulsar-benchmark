# Matched controller tuning and an explicit advantage test

This experiment improves the first Jacobi controller and tests where its
benefits come from. It does not take “Jacobi is better” as an assumption. A
coordinate representation, a guidance policy, a navigation estimator and a
hardware implementation are different potential sources of benefit.

The implementation adds:

- A causal nine-state filter for Moon-relative position, velocity and a slowly
  varying inertial acceleration bias. It uses delayed direct-state fixes with
  their measurement epoch and an explicit command-knowledge cutoff.
- Innovation gating for bad fixes, Joseph covariance updates, process noise
  for uncertain forces, and approximate maneuver execution covariance.
- Bias-aware predictive rollouts and planning. The reference keeps the nominal
  force model; the estimated disturbance enters the spacecraft prediction and
  compensation, so adding it to both sides cannot silently cancel it.
- A matched Moon-centered Cartesian implementation using the same physical
  forces and numerical integrator, plus PD and conventional target-point
  guidance baselines with the same finite-burn checks and actuator constraints.
- A written training/evaluation split, paired sensor and actuator randomness,
  frozen training selection, and a separate month-long scenario.

Files are `qrc_pulsar_bench/adaptive_control.py`,
`qrc_pulsar_bench/control_tuning.py`, and `tests/test_adaptive_control.py`.
The original controller gained two extension points for linearization and
candidate generation; its default behavior and public interface are preserved.

## What changed after the pilot

The first filter pilot was overconfident about changing forces. After an
unmodelled disturbance it rejected valid fixes and allowed the estimated state
to diverge. This was an actual failed test, not a successful robustness result.
The repeated pilot includes random-walk force uncertainty and maneuver
magnitude/pointing covariance. It accepts ordinary fixes and rejects the single
injected corrupt fix. Both pilot receipts are retained; neither pilot is an
evaluation case.

Direct-state measurements are still synthetic. Estimating an acceleration bias
does not identify which force caused it, nor does it close missing attitude,
mass, thermal or sensor-model inputs. Innovation gating cannot distinguish
every coherent sensor fault from a real trajectory change. Its calibration is
reported using innovation and state-error statistics, not assumed from a
successful solve.

## Frozen comparison protocol

The plant is propagated independently in Moon-relative Cartesian coordinates
using the sourced lunar/planetary force model. Every controller receives the
same planned physical orbit, sensor cadence, ten-minute delivery delay,
measurement error draws and physical actuator limits. Future true spacecraft
states never enter the controller. True state is used to generate available
measurements and to compute diagnostics afterward.

The plant additionally receives a declared unmodelled acceleration, a force
episode, thrust magnitude error and thrust pointing error. Actuator error draws
are keyed to decision epochs; taking fewer burns does not change the random
sequence used by another method. Fuel flow is commanded while delivered thrust
is imperfect, so mass accounting stays explicit. Sensor-outage cases withhold
fix deliveries. The outlier is a declared corrupted position/velocity fix, not
a force inferred from a trajectory residual.

Training uses three six-day scenarios, seeds 1101–1103, lunar radii 8,000,
10,000 and 14,000 km, and inclinations 35, 55 and 70 degrees. Each guidance
family gets eight configurations. An initial four-configuration training pass
showed that the target-point baseline also needed a fuel-weight sweep. The grid
was expanded equally for all families using training results only, before any
evaluation cases were opened; both passes are retained:

| Family | Parameters explored |
| --- | --- |
| Predictive | Fuel penalty, one/two-day horizon and hysteresis deadband |
| Damped PD | 6, 8, 10, 12, 18, 24, 36 and 48-hour response time |
| Target point | Targets 6, 12, 24 or 48 hours ahead, plus fuel-weight alternatives at 6/12 hours |

The conventional target-point controller solves a weighted position/velocity
correction using the same state-transition model. It supplies a stronger
comparison than PD alone at a six-hour correction cadence. Similar target-point
and orbit-maintenance studies explicitly account for navigation and maneuver
execution errors; see the [ARTEMIS station-keeping analysis](https://artemis.igpp.ucla.edu/pubs/2010_nonrefereed/AAS_10-113.pdf)
and [NASA's NRHO orbit-determination study](https://ntrs.nasa.gov/citations/20200011550).

Selection reads **training results only**. A candidate must have mean steady
RMS at most 300 m, worst training-scenario steady RMS at most 500 m, maximum
sampled error at most 5 km, and no decision exceeding the declared budget of
5,000 online force evaluations. Among qualifying candidates, minimum average
commanded delta-v wins. If none qualify, the result says so and selects the
smallest worst-case RMS only for further diagnosis. It does not relabel that
candidate as meeting the requirements.

Steady RMS excludes the initial 24-hour acquisition interval. Full-interval RMS,
maxima and final error remain in the receipts. These thresholds and exclusions
are experiment requirements, not mission requirements inferred from NASA data.
The computational criterion is a benchmark eligibility test, not a certified
execution deadline or an enforced flight watchdog.

Evaluation uses six different nine-day scenarios, seeds 2201–2206, radii 7,000,
12,000 and 16,000 km, inclinations 25, 50 and 65 degrees. Cases include stronger
measurement noise, up to 8% thrust dispersion, pointing error, one-day lost-fix
intervals, force mismatch and corrupted fixes. The selected settings remain
frozen. Source and protocol hashes must match training or evaluation refuses
to run. A separate 30-day scenario uses seed 3301 and a two-day lost-fix interval.

The comparison includes the original raw-fix controller, tuned predictive
guidance, tuned PD, tuned target-point guidance, the identical predictive policy
in Cartesian coordinates, and an acceleration-feedforward ablation. The
ablation retains the same filter but does not feed its acceleration estimate
to guidance. It therefore isolates that guidance input, not all effects of
bias estimation inside navigation.

## Accuracy, computation and remaining scope

Both coordinate implementations use the same binary64 force model, DOP853
tolerances and collision guards. The independent plant uses a smaller maximum
step. The earlier ninety-day coordinate control remains numerical evidence;
using the same tolerances alone does not establish a general coordinate
accuracy advantage. The full force/coordinate conventions are in
[the original controller guide](jacobi-predictive-control.md).

Reported process CPU separates guidance and navigation. Reference construction
is offline and excluded for every method. Some independent cases run in
separate processes with separate ephemeris/SPICE state; process CPU is not a
flight processor latency, FPGA resource estimate or power measurement. Force
evaluation counts provide a separate reproducible measure of online work.

The filter integrates its mean under the complete model and uses piecewise
midpoint linearization for covariance. Finite-burn uncertainty is approximated
at burn completion with a half-burn position correlation. Neither is a full
nonlinear uncertainty distribution or coupled attitude/propulsion estimator.
Partial-burn estimation, correlated/biasing sensors, actual sensor observation
models, multiple spacecraft hazards and rapid interrupt-driven safety checks
remain outside this experiment. The legacy emergency supervisor and its
finite-horizon limitations remain unchanged.

All errors and savings are conditional on these synthetic scenarios. Six
evaluation cases cannot establish a flight-failure probability, universal
nonoscillation, fuel savings on arbitrary missions or a unique QRC advantage.

## Reproduction

Use the physical input files and native harmonic library described in
[the original controller guide](jacobi-predictive-control.md). Each phase writes
full measurements-derived estimates, commands, statistics, source hashes and
its frozen protocol:

```sh
python -m qrc_pulsar_bench.control_tuning \
  --physics-dir /path/to/gravity-and-orientation \
  --ephemeris /path/to/de440s.bsp --gm-kernel /path/to/gm_de440.tpc \
  --native-library /tmp/liblunar.so --output /tmp/control-comparison --phase train

# Repeat with the same arguments and --phase evaluate, then --phase month.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The prototype's independent analytic tests cover random-walk covariance,
constant acceleration identification, delayed updates, outlier recovery,
command availability and mass accounting, execution covariance, bias
feedforward and matched-coordinate commands. The original finite-burn,
clearance, force and language-bridge checks remain required.
