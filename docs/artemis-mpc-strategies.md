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
Direction changes between successive executed burns and between adjacent
30-minute burns are distinct reported diagnostics. Neither is a proof of global
stability. Real attitude/thruster allocation, minimum impulse hardware behavior,
fuel uncertainty and mission-specific emergency handling remain unvalidated.

## Reproduction

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

Use a new output directory after numerical-source or protocol changes. Saved
source snapshots and all input hashes allow the executed implementation to be
distinguished from later edits. The scripts require a repository checkout for
the native `hardware/` sources.
