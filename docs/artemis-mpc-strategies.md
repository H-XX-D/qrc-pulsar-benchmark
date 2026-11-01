# ARTEMIS predictive-control strategy search

This experiment searches for low fuel use and infrequent corrections while
keeping the spacecraft close to its planned trajectory. It reports the best
measured tradeoffs among a finite set of strategies; it does not claim a global
optimum over every trajectory, control policy or uncertainty realization.

The reference is propagated from the initial definitive ARTEMIS state. The
controlled vehicle and navigation fixes remain synthetic, with the same generic
500 kg / 0.2 N actuator and disturbance model used in the previous experiment.
Future recorded spacecraft states are not guidance inputs. The new comparison
scores position error every five minutes using the independent plant's dense
trajectory, rather than only at the 30-minute decision epochs. Steady scoring
excludes the first 12 hours. Five-minute maxima are sampled diagnostics, not
certified continuous-time error bounds.

The definitive initial states are reconstructed orbit products and can themselves
depend on later tracking. The January split tests frozen controllers on new
geometries and synthetic noise/fault cases; it is not an operationally available
initial-fix test or validation against the historical spacecraft's reactions to
our commands.

The [executed results](artemis-strategies-results-2026-10-31.md) include every
training candidate, the fresh comparison and individual tracking traces.
The stronger development candidate after evaluation is `sparse_05`: mean
tracking RMS fell 9.73%, delta-v 18.65% and burn count 35.02% versus baseline.
It met the broader 100 m RMS / 500 m peak limits in all eight January cases.
The training winner, `sparse_event60`, saved more fuel but failed those limits
in one outage case. Neither met the tighter 50 m RMS / 300 m peak envelope in
every fresh case. This is a measured tradeoff, not a completed global optimum.

## Strategies compared

The 15 settings include the previous selected MPC baseline and these changes:

- Quadratic penalties on changes between successive planned corrections, using
  two smoothing strengths. The first difference uses the previous accepted
  command slot; coasting or stale navigation clears that slot.
- One- and two-hour spacing between burns. The optimizer removes unavailable
  future burn slots rather than optimizing a dense schedule and blindly
  discarding its first command. A predicted excursion over 500 m can override
  the current dwell gate; this is logged and is not a certified emergency policy.
- Event triggers at predicted 30, 60 or 100 m drift. Inside the threshold, the
  vehicle coasts. These thresholds concern modeled future tracking error, not
  body-clearance guarantees.
- Sparse effort penalties, implemented with six fixed iteratively reweighted
  quadratic solves and an explicit minimum useful command. The penalty
  approximates a sum of vector norms; it is not an exact burn-count optimizer.
- Prediction using the causal filter's estimated acceleration bias, with and
  without a coast trigger. It never reads the plant's hidden injected bias.
- Combined event/spacing and sparse/event strategies, and an uncertainty-aware
  trigger with a two-sigma allowance. That allowance is a heuristic; it is not a
  nonlinear reachable set or an established collision probability bound.

The sparse-control idea is supported by the literature on
[minimum-fuel spacecraft control using iteratively reweighted least squares](https://arxiv.org/abs/2109.08781).
Predictive coast triggers similarly address the problem of unnecessary small
firings in [event-triggered spacecraft hovering control](https://www.sciencedirect.com/science/article/pii/S2405896319310122).
Those papers motivate the choices; their guarantees do not transfer to this
different force model, estimator and actuator without proof.

## Selection and untouched evaluation

The frozen search uses 60 three-day training runs: 15 settings, two September
ARTEMIS geometries, and two noise/fault seeds per geometry. A separate pilot
checks four families. The same sensor, noise draws, force law and finite-burn
errors are used across strategies. Fault cases contain a six-hour navigation
outage and one corrupted fix.

A candidate qualifies only if all four training cases have steady RMS at most
100 m, steady sampled peak at most 500 m, and no collision, solver, arithmetic or
first-burn clearance-screen failure. The report retains the entire search,
including failures. It identifies the Pareto set on worst-case training RMS,
mean delta-v and mean burn count. Direction changes are also reported.

The primary recommendation uses the tighter 50 m RMS / 300 m peak envelope when
any candidate meets it. Within that envelope it minimizes a declared equal-weight
score: worst RMS / 50 m + mean delta-v / baseline delta-v + mean burns / baseline
burns. This is an explicit preference rule, not a mathematical definition of
universal optimality. The tightest, least-fuel and fewest-burn candidates are
also retained so the tradeoffs stay visible.

The primary setting is checked at 24/16 and 32/24 total/fractional precision on
training cases before selecting a width. The gates are 1% relative changes in
RMS and delta-v, 10 m change in steady peak and at most one burn-count change per
case, with no solver/range/screen fault.

Fresh evaluation uses P1/P2 initial states on January 1 and January 10, 2012,
each with ordinary and outage seeds 8101/8102: eight cases per selected policy.
Sixteen definitive V02 CDF files and two DEF SPKs are hashed. January outcomes
do not choose the strategies or precision. The predeclared finalists and baseline
run in floating point; the primary also runs at its selected integer width.

## Hardware and operational boundaries

The existing fixed-work integer QP core is reused. Ordinary policies use two
256-iteration solves per active update; sparse policies use six. Event-based
coasting avoids the QP call, but still computes a model prediction on the host.
Navigation, prediction, bias handling, scheduling, IRLS weights and nonlinear
burn screening are host operations. No board has been selected and there are
no new synthesis, RTL timing, resource or power measurements.

Smaller delta-v is not necessarily fewer corrections or lower pointing cost.
Changes in commanded direction between successive burns and between adjacent
30-minute burns are distinct reported diagnostics; the count does not include
pointing-error perturbations. Neither is a proof of global
stability. Real attitude/thruster allocation, minimum impulse hardware behavior,
fuel uncertainty and mission-specific emergency handling remain unvalidated.

## Reproduction

The frozen primary is `strategies()['sparse_event60']`, rather than the
illustrative defaults of `MPCSettings()`. It uses four 30-minute prediction
slots, a 60 m coast trigger, sparse weight 0.1 and a 0.001 m/s minimum requested
correction. Sparse effort and finite-burn timing use six fixed refinements.
The 24-bit / 16-fractional-bit QP core failed the 1% tracking-change gate in
one training case (5.515% RMS change). The 32-bit / 24-fractional-bit core
passed all four, with maximum RMS change 0.003655% and delta-v change 0.001394%.
The selected format applies to this policy; the previous baseline's 24-bit
result remains a separate measurement.

For the stronger development candidate, use `strategies()['sparse_05']`:
the same two-hour horizon, sparse weight 0.5, minimum correction 0.001 m/s,
and no coast trigger or mandatory burn spacing. An additional 12 arithmetic
runs checked its frozen settings at 32 total / 24 fractional bits. All passed;
the largest RMS change was 0.011955%, largest delta-v change 0.003693%, and
every burn count matched floating point. These checks reused the September and
January cases after evaluation, so they are not new independent holdout data.
The full campaign totals 116 three-day runs, including pilots and arithmetic
comparisons, across the 15 settings. The complete test suite passed 349 tests.

```sh
python -m pip install -e '.[navigation,test]' matplotlib
PYTHONPATH=. python scripts/artemis_strategy_campaign.py prepare \
  --output /tmp/artemis-strategies --base /path/to/artemis-pid-campaign \
  --kernels /path/to/planetary-kernels --physics /path/to/lunar-inputs \
  --native /path/to/liblunar.so
# Repeat the campaign command with pilot, train, precision, then evaluate.
python scripts/report_artemis_strategies.py /tmp/artemis-strategies
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The executed campaign is stored at
`/Users/hendrixx./qrc-mpc-strategies-2026-10-31/`. Its
`selected-controller.json` exports the selected settings, arithmetic and evidence
receipts. The `verification.json` receipt checks all 65 numerical source files
and their saved copies, all 53 physical inputs, selection provenance, complete
run matrices, independently recomputed metrics and the packaged modules.
`scripts/verify_artemis_strategies.py` recreates those final receipts after the
campaign, wheel build, objective audit and test log are present.

Use a new output directory after numerical-source or protocol changes. Saved
source snapshots and all input hashes allow the executed implementation to be
distinguished from later edits. The scripts require a repository checkout for
the native `hardware/` sources.
