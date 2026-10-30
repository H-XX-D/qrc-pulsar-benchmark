# Cislunar onboard navigation target

The author clarified on 2026-10-30 that the intended application is onboard
orbit estimation and prediction, reducing dependence on corrections computed
on Earth, and selected Earth–Moon/cislunar spacecraft as the first target.

This document specifies the next benchmark. It records no executed cislunar
navigation result and makes no flight-readiness claim. The existing J0337
experiments remain useful mechanics and numerical checks; their timing residuals
are not spacecraft position errors or a measure of navigation autonomy.

## Operational question

Given an uncertain initial state, a preloaded force model and ephemerides, and
only measurements available aboard the spacecraft, can a relational/shape
formulation maintain a useful orbit estimate with less computation, memory or
external assistance than a conventional implementation at the same accuracy?

Separate three capabilities:

1. Propagation: advance an estimated state and its uncertainty between observations.
2. Orbit determination: update that estimate from noisy observations as they arrive.
3. Guidance/control: choose and execute maneuvers from the estimated state, a
   desired trajectory and spacecraft constraints.

The first benchmark covers propagation and orbit determination. Maneuver
execution is a later closed-loop test. A change of coordinates alone cannot
observe an unknown disturbance or remove uncertainty in the starting state.
Onboard measurement updates count as autonomy; receiving ground-computed state
resets must be recorded separately.

## Comparison protocol

Use an Earth–Moon–spacecraft case with spacecraft gravity negligible. Begin with
the restricted three-body problem for a controlled coordinate comparison. Add
ephemeris-driven Earth/Moon/Sun forces, nonspherical gravity, solar radiation
pressure and known maneuvers in successive fidelity levels. The circular
restricted model is not sufficient by itself to validate a real lunar trajectory.

Compare a conventional Cartesian or Jacobi propagator with direct shape
evolution. Use the same physical initial-state distribution, force model,
observations, measurement covariance and estimator update policy. Transform
covariances and measurement Jacobians consistently between representations.
Equal numerical tolerances in different coordinates do not establish equal
physical error; compare cost at matched reconstructed position/velocity error.

Use sequential filtering rather than a batch fit that sees the future. Score
both prediction between observations and estimation after observation updates.
Do not allow one implementation better measurements or more frequent resets.

The current mass-weighted planar shape chart needs an explicit treatment of the
negligible-spacecraft-mass limit: its second Jacobi weight tends to zero. The
spatial triangle frame also needs near-collinearity tests and, if necessary,
alternative coordinate charts. Preserve physical scale, orientation, velocity
and time reconstruction. The shape sphere alone is not a complete orbit state.

## Measurements and data provenance

The sensor suite has not been selected. Plausible configurations include optical
bearings to the Earth/Moon or mapped landmarks, spacecraft crosslink range and
range rate, and weak-signal GNSS where available. A star tracker principally
constrains attitude; an orbit measurement must have a specified relation to
position, velocity or clock. Clock errors and reference-spacecraft uncertainty
belong in the estimator when relevant. Check observability for the chosen
geometry before claiming that all state components can be estimated.

Use two explicitly labeled evidence levels:

- Simulation: independently generated trajectories and synthetic sensor errors,
  including force-model mismatch. Real planetary ephemerides do not turn
  synthetic spacecraft observations into flight measurements.
- Flight-data replay: archived actual tracking/sensor observations processed in
  time order, with an independent reconstructed orbit used only for scoring.
  Verify archive availability, uncertainty, reference frame, time scale and
  maneuver history before selecting the case. CAPSTONE/LRO are relevant mission
  examples, not a claim that their raw navigation observations have been acquired.

Preloaded Earth/Moon ephemerides are permitted and must be versioned. Later
spacecraft trajectory samples, post-fit corrections and the scoring orbit must
never enter propagation or measurement updates. A reconstructed SPICE trajectory
is a reference product; interpolating it is not independent onboard navigation.

## Tests and reporting

Measure errors and cost over several initial-state perturbations and noise
realizations, with seeds fixed before comparing methods. Include ordinary
tracking, onboard sensor outages, loss of ground contact, measurement biases,
near-collinear geometry, close lunar passage and maneuver execution errors.
Loss of ground contact is distinct from loss of all onboard measurements.

Report:

| Quantity | Purpose |
|---|---|
| 3D and radial/along-track/cross-track position error, metres | Orbit knowledge |
| Velocity error, metres per second | Prediction and maneuver relevance |
| RMS, 95th percentile and maximum error | Typical and worst sampled behavior |
| Forecast error after 1, 6, 24 and 72 hours | Prediction horizon, subject to trajectory coverage |
| Covariance coverage and innovation consistency | Whether confidence matches observed errors |
| Recovery time after measurement outage | Robustness of sequential estimation |
| Number and source of external state resets | Dependence on ground assistance |
| Per-cycle latency, memory and force evaluations | Compute cost at matched accuracy |
| Energy per cycle on specified hardware, if measured | Power cost; do not infer from laptop time |

The error requirement, update rate, acceptable outage duration, processor and
sensor specifications remain mission inputs. Explore an accuracy/cost tradeoff
until these are fixed; do not invent a flight acceptance threshold. A useful
result could be lower compute at equal error, longer tolerable outages, or better
calibrated uncertainty. A coordinate transformation or favorable plot alone
does not establish any of those gains.

## Primary references

- [NASA GEONS](https://etd.gsfc.nasa.gov/capabilities/capabilities-listing/geons/):
  an existing onboard navigation baseline using measurements, dynamics and an EKF.
- [NASA autoNGC technical presentation](https://ntrs.nasa.gov/citations/20240002710):
  lunar navigation architecture and sensor-dependent performance analysis.
- [Autonomous Crosslink Radionavigation for a Lunar CubeSat Mission](https://arxiv.org/abs/2204.14155):
  simulated lunar crosslink estimation and observability analysis.
- [NASA CAPSTONE](https://www.nasa.gov/smallspacecraft/capstone/):
  cislunar crosslink navigation with LRO.
- [NAIF SPICE concept](https://naif.jpl.nasa.gov/naif/spiceconcept.html):
  geometry and ancillary data products, distinct from raw sensor measurements.
