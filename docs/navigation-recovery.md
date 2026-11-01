# Square-root navigation and velocity-change reacquisition

The mission objective is ground-supervised autonomy with survival during an
extended loss of ground contact. Ground may authorize autonomous mission
execution and remain an observer, with the ability to review or intervene.
The spacecraft should detect anomalies, take bounded protective action, adjust
within its authority, and notify ground; a blackout must not make protection
depend on a new ground command. Preserving a usable state estimate, clearance,
resources and recoverability takes priority over tight tracking of the original path.
Ground-link loss is distinct from loss of onboard navigation. These experiments
assume synthetic direct-state fixes remain available except for declared sensor
outages; they do not implement the physical onboard sensors that produce them.

The intended operating contract separates fast protective response from slower
state confirmation and recovery planning. The current experiment supplies fixes
every 30 minutes and requires an hour of confirmation history: an isolated jump
is first observable after 30 minutes and normally confirmed after 90 minutes.
This does not establish a fast emergency-response capability. The bounded
in-memory confirmation log is diagnostic evidence, not a persistent, acknowledged
ground-notification queue. Faster independent protection, durable event downlink,
ground intervention/handback and a verified survival maneuver planner remain
system-integration requirements beyond this navigation repair.

`ManeuverAwareOrbitFilter` adds a provisional-track recovery layer to the causal
nine-state orbit filter. `CausalOrbitFilter` now retains a square-root covariance
internally. The recovery wrapper is opt-in; the navigation campaign exercises it
with the existing `RoutineMPC`. No MPC authority limits have been raised.

## Why the previous covariance failed

The archived P2 0.5 m/s impulse replay fails at decision time 554,400 s (154 h).
After prolonged observation rejection, the predicted covariance is nearly rank
one, with position variances up to 6.88105e19 m². In the full-matrix innovation
sum, five of the six measurement variances disappear at float64 precision.
Solving that rounded matrix returns NIS = -503.186075. The old test `NIS <= 35`
accepts this impossible negative statistic, and the Joseph update produces a
velocity variance of -1.72950e-5 m²/s². The update commits before checking it.

Passive replay instrumentation found no invalid process-noise covariance. This
isolates the observed numerical trigger to loss of covariance information and
the resulting innovation/update arithmetic, following prolonged rejection.
It does not establish that all long-gap EKF linearizations are physically valid.
Reproduce the archived failure with `scripts/diagnose_navigation_covariance.py`.

## Covariance representation

The persistent numerical state is a factor `L` such that `P = L Lᵀ`. Prediction
triangularizes the columns `[Φ L, L_Q]` by QR; finite-burn execution uncertainty
is incorporated as another set of columns. Measurement conditioning performs
QR on the transpose of the joint array

```text
[ L_R    H L ]
[  0      L  ]
```

Its lower-triangular result supplies the innovation factor, state correction,
and conditional posterior factor. NIS is the squared norm of the innovation
whitened by a triangular solve. There is no full-matrix `P + R` solve and no
subtraction of nearly equal covariances. This is standard square-root filtering,
not a new orbital theory; see the [JPL derivation](https://ipnpr.jpl.nasa.gov/progress_report/42-233/42-233A-orig.pdf).

`covariance` is a derived reporting view; editing the returned array does not
alter the filter. The live factor is never recovered from that rounded view.
Input and process covariances are factored in correlation units. Semidefinite
inputs permit only a dimension-scaled machine-roundoff eigenvalue tolerance;
the code does not repair a damaged live covariance by clipping its eigenvalues.
State, factor, timestamps and counters commit only after validation succeeds.
Nonfinite or invalid calculations raise, preserving the last committed estimate.

## Causal reacquisition

The recovery wrapper accepts the same delayed fixes and known-command ledger as
the base filter. It never receives truth states, anomaly labels, kick times or
fault directions.

1. A rejected fix marks navigation suspect. A velocity discrepancy of 0.03–2 m/s
   with velocity NIS at least 25 may seed one provisional track. The initial
   position discrepancy must be within 5 km, and the last accepted observation
   must be within three hours.
2. The provisional track starts from that fix, with four times its measurement
   covariance and a fresh acceleration-bias prior. The seed cannot count twice.
3. Later fixes must fit the propagated candidate, including known finite burns,
   with six-component NIS at most 16.8. A failed confirmation or gap longer than
   an hour discards that candidate. Replayed or reversed epochs are rejected.
4. Promotion requires at least three fixes spanning an hour, with largest-axis
   three-sigma bounds at most 200 m and 0.02 m/s. The candidate replaces the old
   track; the correlated tracks are not fused as independent evidence.
5. The forecast reports `navigation_trusted`, `navigation_mode` and
   `candidate_confirmations`. `RoutineMPC` inhibits untrusted navigation and
   refuses rearming from it. A confirmed estimate still has to satisfy the
   existing tracking, clearance, cadence, actuator and fuel limits.

A separate path handles a real gap of more than three hours in **received**
measurements. When the returning fixes fail the ordinary innovation gate, the
old track may be too stale to constrain a reset. A provisional track can then
start from the measured state without the old-track position/jump limits; it
must still pass the same three-fix dynamic-consistency and uncertainty checks.
Rejection of the first returning fix does not disable subsequent attempts.
A continuous sequence of rejected fixes cannot create this missing-data mode.
Its event reason is `observation_gap`, not a physical-impulse attribution.

This path repairs an observed V2 miss: a three-hour sensor outage left the last
received fix 3.5 hours old on return. The recent-track detector refused to seed,
and ordinary filtering did not regain an accepted fix until 100.5 hours after
injection. V2 is retained as evidence; V3 adds fresh noise/direction instances
with six-hour outages and inconsistent sensor data after an outage.

The event log retains at most 128 confirmations and only one candidate exists
at a time. The fixed state dimension and bounded history do not establish a
worst-case execution time for the adaptive ODE or a synthesized FPGA design.

```python
from qrc_pulsar_bench.navigation_recovery import ManeuverAwareOrbitFilter

nav = ManeuverAwareOrbitFilter(model, initial_state, initial_covariance, mass)
nav.observe(fix, fix_covariance, measurement_s, available_s, known_commands)
estimate = nav.forecast(decision_s, known_commands, as_of_s=decision_s)
dv = routine_mpc.command(decision_s, estimate, estimate['mass'])
```

## Meaning of confirmation

A confirmed alternative track establishes consistency with the supplied sensor
stream and force model. It does not prove that an impulse physically occurred.
For example, clearing a biased sensor can require replacing a damaged estimate;
that is navigation recovery without a new physical maneuver. A coordinated
sensor fault that reports a plausible changed trajectory is indistinguishable
from actual motion in this single direct-state stream. Tests explicitly retain
that counterexample. Independent navigation observables are required to resolve
it, and the synthetic Gaussian measurement assumptions need flight validation.

Positive covariance alone is also insufficient evidence of statistical
calibration. The EKF still uses local linearization, approximate burn uncertainty
and a clipped bias estimate. The campaign reports empirical navigation errors
against the declared bounds and separately scores trajectory return.

Results and reproducibility checks are in
[the navigation campaign report](artemis-navigation-results-2026-10-31.md).
Only the 32/24 MPC QP core runs in integer host emulation; navigation QR remains
float64. FPGA synthesis, radiation tolerance, execution timing and flight
qualification remain untested in this implementation.
