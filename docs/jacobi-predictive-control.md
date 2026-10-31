# Jacobi guidance with a mission reference, damped feedback and emergency search

This is an executable **simulation controller** for a spacecraft near the Moon.
It combines a long mission reference with shorter correction plans. Its force
interface includes the recent velocity-dependent and planetary perturbations.
It is not flight-qualified software, a demonstrated raw-sensor navigation
system, a certified collision-avoidance controller, or an FPGA benchmark.

## State, inputs and reference discipline

Let `r1 = r_Moon - r_Earth` be the inner Jacobi vector and
`mu = GM_Moon/(GM_Earth + GM_Moon)`. The integrated outer vector is

```
q = r_spacecraft_relative_to_Moon + (1 - mu) r1
q_dot = v_spacecraft_relative_to_Moon + (1 - mu) r1_dot
q_ddot = a_relative_to_Moon(r, v, t) + (1 - mu) r1_ddot + thrust/mass
```

The inner orbit is prescribed by a preloaded planetary ephemeris. The outer
orbit is integrated directly in Jacobi coordinates. This is a restricted,
ephemeris-driven model, not an independently evolved twelve-component
Earth–Moon–spacecraft system and not direct shape-sphere evolution.
Position, velocity and acceleration of the coordinate origin come from the
same spline; simply translating positions without translating acceleration
would introduce a fictitious force.

The demonstration uses DE440s, GRGM900C lunar harmonics, DE421 lunar orientation,
Earth and Sun differential attraction, Earth J2, additional planetary-system
tides, full point-mass first-PN corrections, and eclipse-modulated cannonball
solar pressure. Lunar elastic tides remain excluded because the gravity
product's permanent-tide convention is unresolved. The existing
[force budget](cislunar-error-budget-and-advantage.md) still applies; adding a
controller does not close the spacecraft-telemetry gaps.

The guidance API receives only a `NavigationEstimate`, current mass, the force
model, actuator limits and a planned reference. A reference may contain future
**commanded target states**. It must not be replaced by future observed states.
The example constructs its entire reference from a declared initial condition
before applying feedback. No spacecraft archive or differentiated residual is
read by the controller or the demonstration.

The demonstration's observations are explicitly synthetic noisy position and
velocity fixes. They arrive ten minutes after their measurement epoch and are
propagated to the decision epoch with the model and the accepted command log.
The true trajectory is used only to generate measurements at available epochs
and to score results. This is not an implementation of an optical sensor,
GNSS receiver, pulsar TOA pipeline or real navigation-data ingestion.

## Planning and damping

The long plan spans 90 days. At each six-hour decision epoch the controller:

1. Predicts a no-burn trajectory over the next decision interval and checks
   Earth and Moon clearance.
2. Checks observation age, hysteresis deadbands, fuel reserve and burn spacing.
3. Linearizes the complete force callback, including derivatives with respect
   to velocity, around the reference. It plans advisory impulses over the next
   two days using constrained least squares and a quadratic fuel penalty.
4. Constructs a critically damped PD candidate as an alternative and fallback.
5. Converts each first impulse into a finite burn with changing mass. It
   evaluates the actual nonlinear trajectory before selecting a command.
6. Executes only the selected first burn; the later plan is recomputed from
   subsequent navigation estimates.

Thus the campaign covers months, while individual corrections are planned on a
rolling two-day horizon. It does not freeze a three-month burn schedule or
claim globally optimal fuel use. NASA's
[NRHO maintenance study](https://ntrs.nasa.gov/api/citations/20170001347/downloads/20170001347.pdf)
likewise separates a long reference from shorter maintenance targeting. The
example here uses a high lunar orbit and has not validated an NRHO controller.

For error `e = estimated_state - reference_state`, the PD acceleration is

```
u = a(reference) - a(estimate) - e_position/tau² - 2 e_velocity/tau
```

With exact, continuous, unsaturated actuation and correct state knowledge this
gives `e_position'' + 2 e_position'/tau + e_position/tau² = 0`.
The repeated real pole `-1/tau` gives critical damping. Initial position error
with zero velocity error decays monotonically in the independent analytic
test. Arbitrary initial velocities can cross the reference; finite burns,
discrete measurements, model errors and saturation do not inherit a universal
nonoscillation theorem.

Position and velocity deadbands have separate on/off thresholds. Routine burns
must be at least six hours apart. There is no integral accumulator to wind up
during saturation or loss of navigation. Stale observations inhibit routine
burns. A failed numerical planner falls back to the independently validated PD
candidate. Planning alone does not consume fuel or update the burn-spacing
state; `confirm_execution` records acceptance of a command.

## Emergency behavior and its limits

The supervisor checks predicted clearance before normal tracking logic. When
coasting predicts entry into an exclusion zone, it tests a finite set of
bounded burns in radial, tangential, Earth-relative and inertial directions.
It selects the smallest tested delta-v that maintains predicted clearance
through the decision horizon. Emergency burns may use the reserved propellant
and bypass routine burn spacing.

The status distinguishes `EMERGENCY_BURN`, `NO_FEASIBLE_EMERGENCY`,
`CLEARANCE_ALREADY_VIOLATED`, `NAVIGATION_STALE`, and
`ROUTINE_AUTHORITY_EXHAUSTED`. A fixed prograde burn is not assumed to be a
rescue. Failure to find a burn does not prove that no possible maneuver exists;
the search is finite. An estimate already inside the inflated exclusion zone
is reported for intervention, not declared recovered.

Safety margins use supplied position/velocity uncertainty with a declared
linear inflation heuristic. This is not a propagated nonlinear reachable set
or a calibrated collision probability. Integrator events detect exclusion-zone
crossings; accepted nodes and midpoints measure minimum clearance. Sub-step
grazing encounters are not formally certified. A six-hour survival check is not
proof of indefinite viability, and a later feasible recovery is not required
as a terminal constraint.

Thrust is assumed available in an arbitrary inertial direction at command
time. Attitude slew, plume restrictions, power limits, ignition delay,
minimum impulse-bit hardware, thermal limits, occulted communications, terrain,
other spacecraft/debris, sensor integrity and individual thruster failures need
mission-specific models. Propellant and dry mass are explicitly bounded, but
the assumed solar-pressure area/mass ratio is fixed during these small burns.
The controller cannot guarantee survival after loss of propulsion or inadequate
warning time. No risk percentage or reduction in required ground contacts is
inferred from these simulations.

## Reproduction

The source is split into the controller, a synthetic experiment and an
independent numerical/emergency check:

- `qrc_pulsar_bench/jacobi_control.py`
- `qrc_pulsar_bench/jacobi_control_demo.py`
- `qrc_pulsar_bench/jacobi_control_validation.py`
- `tests/test_jacobi_control.py`

Use the same downloaded inputs described in
[the gravity refinement guide](cislunar-force-refinement.md). Build the existing
host C++ harmonic evaluator; this does not synthesize or benchmark an FPGA.

```sh
clang++ -std=c++17 -O2 -shared -fPIC hardware/lunar_harmonics.cpp -o /tmp/liblunar.so

python -m qrc_pulsar_bench.jacobi_control_demo \
  --physics-dir /path/to/gravity-and-orientation \
  --ephemeris /path/to/de440s.bsp --gm-kernel /path/to/gm_de440.tpc \
  --native-library /tmp/liblunar.so --output /tmp/jacobi-control --days 90

python -m qrc_pulsar_bench.jacobi_control_validation \
  --physics-dir /path/to/gravity-and-orientation \
  --ephemeris /path/to/de440s.bsp --gm-kernel /path/to/gm_de440.tpc \
  --native-library /tmp/liblunar.so --output /tmp/jacobi-validation.json --days 90

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

Default scenario assumptions are 500 kg initial mass, 480 kg dry mass, 0.5 kg
emergency reserve, 0.2 N routine thrust, 220 s specific impulse, 600 s routine
burn cap and 3,600 s emergency cap. The reference starts at 10,000 km lunar
radius with 45-degree inclination. Measurement errors are independent 30 m
position and 2 mm/s velocity errors per axis. The truth receives an unmodelled
2e-9 m/s² transverse force and 1e-9 m/s² radial force, plus a 2e-7 m/s²
transverse episode on days 30–33. A separate run loses fixes on days 45–47.
These are assumptions, not fitted spacecraft properties or measured fault rates.

## Numerical verification and interpretation

The first 90-day coordinate comparison failed: a C2 cubic origin interpolation
and looser integration differed by 581.35 m from an independently integrated
Moon-centered control. That error was too large relative to the tracking
target. The origin is now a smoother quintic spline and the Jacobi integration
uses tighter tolerances. The repeated comparison differs by at most 0.626 m
over 90 days, including all modeled forces. This is a measured numerical
agreement, not a proof of physical accuracy. The first, superseded feedback
campaign was stopped and retained separately for audit.

Degree-20 versus degree-120 lunar gravity differs by less than 5e-22 m/s² on
the tested high-orbit vectors. This supports the degree-20 choice for this
campaign's geometry; it does not support that truncation near the lunar surface.
The emergency stress case also uses degree 20 and is a control logic experiment,
not a high-fidelity low-lunar mission analysis.

The native gravity contract remains the existing binary64 C ABI, with caller-owned
buffers and explicit status returns. Its independent normalization, derivative,
pole and invalid-input tests remain part of the full suite. No new language
bridge or hardware completion claim is introduced.

The Recall Gate-3 tripwire was expanded and its capped graph continued from its
boundary. The relevant orbital branches were followed separately. Current
controller/force imports and the native bridge were inspected; quantum OPT/BFV
completion assumptions do not enter these ODEs. The broader theoretical graph
remains capped and is not claimed to have been re-proved. The frozen ARTEMIS
forecast and pulsar timing experiment were not modified.
