# Predictive MPC for the ARTEMIS control experiment

This implementation replaces PID guidance with a receding-horizon optimization.
The previously tuned one-hour PID remains a comparison baseline. The MPC path
does not call a PID controller, select a PD candidate or fall back to PD after
an optimizer failure.

## What is predicted and optimized

At each 30-minute navigation update, the planner linearizes the full force model
along a nominal trajectory generated from the initial spacecraft state. Future
recorded spacecraft positions do not define that reference. Restricted Jacobi
and Moon-relative errors are equal under their common prescribed translation;
this controller does not claim a coordinate-based improvement.

The stage model is `e[k+1] = Phi[k] e[k] + B[k] delta_v[k]`.
The objective sums scaled position/velocity tracking error and a quadratic
delta-v penalty across the horizon. This penalty is a fuel proxy, not exact
minimum propellant. All scheduled impulses obey conservative component bounds
inside the actuator's vector limit and a horizon-wide routine fuel allowance.
The emergency reserve is excluded from that allowance.

The optimizer solves a bounded quadratic program and returns a sequence of
future corrections. Only its first correction is eligible for execution; the
next navigation update produces a new plan. This follows the usual constrained
[receding-horizon MPC structure](https://osqp.org/docs/examples/mpc.html).

Each update uses two fixed-budget solves. The first estimates an impulse plan;
the second incorporates the predicted durations of its early rectangular burns.
The finite-burn linear response uses delta-v units and reduces to the analytic
double-integrator position response `(interval - burn_duration/2) * delta_v`.
Gravity remains locally linearized during each interval. Before execution, the
first actual finite burn is propagated through the nonlinear model and checked
against Moon/Earth clearance floors. Future burns remain tentative.

Solver failure, invalid arithmetic, stale navigation or rejection by the first-
burn screen inhibit routine thrust. That is an explicit failure response, not
a guarantee that coasting is safe. A mission emergency supervisor still needs
separate validation. The present QP constrains thrust/fuel, not a certified
nonlinear obstacle-avoidance tube. Long-term mission planning can supply the
reference; a two- or six-hour receding horizon does not establish months-ahead
robustness or sparse maneuver scheduling.

## FPGA boundary

`hardware/orbit_mpc_core.cpp` is a portable integer QP solver with storage for
36 decision variables (12 three-axis corrections). It performs exactly 256
accelerated projected-gradient iterations per call, checks the final projected
optimality residual and publishes a solution only on success. Each decision
uses at most two calls. It has no heap allocation or convergence-driven loop.

The host constructs a symmetric positive-definite QP, normalizes the maximum
absolute row sum below one, and supplies acceleration and stopping parameters.
The integer core validates dimensions, symmetry, row bounds and input ranges;
positive definiteness is the host builder's responsibility. Dot products use
64-bit accumulators and explicit rounding to nearest with ties away from zero.
The row bound and bounded extrapolated iterate protect the accumulation range.
Failed calls publish zero output. No PID fallback is hidden in the bridge.

Twenty-four- and 32-bit arithmetic are compared on training cases before
evaluation. Both passed for the selected controller; the frozen format is
24 total signed bits with 16 fractional bits. Its maximum relative training
RMS change is 0.2323%, and maximum relative delta-v change is 0.1924%.
The [configuration artifact](artemis-mpc-config-2026-10-31.json) specifies
settings, scaling, matrix layout and status codes. Unlike a fixed gain vector,
the MPC Hessian and gradient are rebuilt for the current state and horizon.
A Python unlimited-integer oracle checks native arithmetic, while
SciPy's bounded least-squares solver supplies an independent optimizer check.
Those checks cover host behavior. No synthesis, RTL, resource, frequency or
power measurement is implied. The HLS script requires an explicit `FPGA_PART`;
hardware has not been selected.

Navigation, force Jacobians, QP formation and the nonlinear first-burn screen
still run on the host. The bounded QP core is the FPGA candidate, not the entire
autonomy stack. The C ABI stores all formats in int32 buffers, so arithmetic
precision alone does not establish hardware resource savings.

## Campaign and evidence

The campaign reuses the exact ARTEMIS synthetic plant, navigation filter,
disturbances, finite imperfect burns and random-draw schedule of the PID study.
Both controllers receive the same state information. Neither gets the filter's
estimated acceleration bias as an extra feedforward input in this comparison.
Navigation observations are synthetic direct-state fixes, not spacecraft
telemetry. The generic 500 kg, 0.2 N actuator is not ARTEMIS's real hardware.

Training uses P1/P2 September initial states and seeds 4101/4102. Six candidates
span two- and six-hour horizons and three control-effort weights. The frozen
rule chooses minimum mean delta-v only among candidates whose four runs have
steady RMS at most 100 m, sampled maximum at most 10 km, no collision and no
optimizer, arithmetic or clearance-screen failure. Failure to qualify is
reported rather than silently selecting a failing configuration.

Precision selection checks the chosen configuration on those same training
cases: at most 1% relative RMS and delta-v differences, at most 10 m change in
sampled maximum error, and no solver/range/screen faults. Fresh evaluation uses
December 1, 2011 P1/P2 initial states and seeds 7101/7102, with ordinary and
six-hour outage/corrupt-fix cases. Eight definitive December V02 CDF files and
the two DEF SPKs have hashed acquisition receipts. No December results select
the horizon, weights or arithmetic width.

The [executed results](artemis-mpc-results-2026-10-31.json) and
[comparison plot](artemis-mpc-results-2026-10-31.png) retain the measured tradeoff.
Full states, planned sequences, commands and source/input hashes are retained
in `/Users/hendrixx./qrc-artemis-mpc-2026-10-31/`.

All 24 floating-point training runs completed. The selected configuration is
`mpc_2h_dv0.05`: a two-hour horizon with four three-axis corrections, 100 m
position and 0.01 m/s velocity cost scales, and a 0.05 m/s effort cost scale.
Its worst training steady RMS is 47.62 m and mean three-day delta-v is
0.35926 m/s. The other qualifying configuration, `mpc_2h_dv0.1`, reaches
48.88 m worst RMS and 0.37734 m/s mean delta-v.

A posthoc fairness audit of the previous September PID training grid applied
the same 100 m RMS gate. It also selects the already-frozen one-hour, zero-
integral PID as the minimum-delta-v qualifying PID. No baseline was changed
after viewing December results. This supports a comparison of these two tuned
controllers; neither finite search establishes a globally optimal controller.

The two-hour setting with the strongest penalty on corrections, and all three
six-hour configurations,
failed the P2 outage scenario, with 3.53–4.47 km steady RMS. Returning fixes were
rejected and routine guidance stayed inhibited; two configurations also recorded
one QP iteration-budget failure. These outcomes are retained. A longer horizon
does not establish greater robustness, and successful optimization does not
remove a navigation-recovery problem. The current evidence supports selecting
the passing configuration, not a claim of global stability.
All six configurations use the same 0.15 m/s vector command cap; the cost weight
changes their willingness to use thrust, not the actuator's physical authority.

An independent audit solved 36 representative orbital QPs with SciPy bounded
least squares. For the selected setting, the largest first-command discrepancy
was below 6e-15 m/s for the fixed-budget floating-point solver, 1.84e-5 m/s for
24-bit arithmetic, and 2.41e-7 m/s for 32 bits. This check uses explicit perturbed
states on training geometry; it does not select gains or constitute closed-loop
flight validation. The full 342-test suite passed, and both arithmetic widths
passed host address/undefined-behavior sanitizers. A previous PID case reproduced
its entire state-error/command traces exactly through the shared harness.

## Fresh December results

The frozen experiment completed 45 control simulations: one pilot, 24 training,
eight precision checks and 12 December evaluations. One additional regression
replayed a previous PID trace exactly. An independent audit compared 36 orbital
QP instances using each of three arithmetic implementations.

| Three-day December scenario | Tuned PID RMS | MPC float RMS | MPC 24-bit RMS | PID / MPC 24-bit delta-v |
| --- | ---: | ---: | ---: | ---: |
| P1 ordinary | 19.72 m | 20.19 m | 20.18 m | 0.6536 / 0.4202 m/s |
| P1 outage and bad fix | 30.81 m | 32.43 m | 32.44 m | 0.7565 / 0.5363 m/s |
| P2 ordinary | 18.06 m | 19.09 m | 19.12 m | 0.5505 / 0.3071 m/s |
| P2 outage and bad fix | 19.46 m | 23.06 m | 22.99 m | 0.7634 / 0.3640 m/s |

Steady RMS excludes the first 12 hours; maximum full-run sampled error is
603.78 m, including the initial offset. The selected MPC has no optimizer,
arithmetic or clearance-screen failures in December. Each faulty case rejects
the injected corrupted fix and inhibits 11 commands during stale navigation.
The largest fixed-versus-float RMS difference is 0.073 m.

Compared in floating point, mean three-day commanded delta-v falls from
0.68098 to 0.40603 m/s (**40.38% lower**), while mean steady RMS rises from
22.013 to 23.693 m (**7.63% higher**). The 24-bit version averages
23.682 m and 0.40690 m/s. Both controllers meet the declared 100 m accuracy
envelope, but the PID has lower RMS in each case. This supports a measured
fuel/accuracy tradeoff for the tested controllers, not universal MPC superiority.

Host online CPU, including navigation, increases from 17.27 to 25.78 seconds
per simulated three days (**49.27%**). The integer-host MPC takes 25.51 seconds.
These process-CPU measurements exclude initial reference construction and are
not FPGA latency or energy estimates. The full physical/navigation stack remains
the dominant integration responsibility outside the integer optimizer.

Both controllers average 138.5 burns per case; the fuel benefit comes from
smaller corrections, not fewer firings. A posthoc direction diagnostic counts
successive executed corrections separated by 30 minutes whose vectors differ
by more than 90 degrees: means are 40.25 for PID, 70.0 for floating MPC and
68.5 for fixed MPC. Such changes do not alone prove unstable orbital motion,
but they prevent claiming that this MPC suppresses alternating commands.
There is currently no explicit slew-rate penalty or dwell constraint in its QP.
Real thruster allocation, pointing cost and wear could change the mission-level
tradeoff. Global nonoscillation, months-long robustness and emergency mission
survival remain unestablished.

## Reproduction

From the repository, with the navigation dependencies and native lunar force
library installed, run the following phases in order. Use a new output directory
when numerical sources or the protocol change.

```sh
python -m pip install -e '.[navigation,test]' matplotlib
PYTHONPATH=. python scripts/artemis_mpc_campaign.py prepare \
  --output /tmp/artemis-mpc --base /path/to/artemis-pid-campaign \
  --kernels /path/to/planetary-kernels --physics /path/to/lunar-inputs \
  --native /path/to/liblunar.so
# Repeat with phases pilot, train, precision.
python scripts/report_artemis_mpc.py /tmp/artemis-mpc --select-precision
# Repeat the campaign command with phase evaluate.
python scripts/report_artemis_mpc.py /tmp/artemis-mpc
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```
