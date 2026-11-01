# Compound faults and ground-supervised autonomy

The operating intent is ground-authorized autonomy: the spacecraft detects and
responds to problems within its assigned authority, reports what it observed and
did, and remains available for ground intervention. During a communications
blackout, the same onboard protection continues. Ground-link availability and
onboard navigation availability are separate inputs.

The [twenty-scenario results](artemis-compound-results-2026-10-31.md) test this
direction with 10 individual faults, 5 simultaneous combinations and 5 ordered
sequences. Each schedule runs for three days from both ARTEMIS P1 and P2 initial
geometries, using float and integer MPC, matched coasting comparisons, and clean
controls. Fault schedules and noise seeds were frozen before execution. The
controller is not retuned between failures or given fault labels.

## What is implemented

- `NavigationIngress` rejects malformed, nonfinite, future-dated and replayed
  measurement packets before they can overwrite the filter. Routine thrust is
  inhibited while the input is invalid. A subsequent valid packet still has to
  pass the filter's ordinary consistency or reacquisition checks. Numerical
  exceptions from the filter are surfaced separately from input rejection.
- `ManeuverAwareOrbitFilter` retains the square-root covariance repair and
  provisional-track checks from the navigation campaign. One candidate must
  accumulate three distinct, dynamically consistent fixes spanning at least
  an hour. A real observation gap can activate cold reacquisition. The compound
  campaign deliberately retains this policy to measure its remaining failures.
- `RoutineMPC` retains its 5 km / 0.5 m/s routine error envelope, navigation-age
  check, nonlinear burn screening and 1 m/s rolling daily issued-command budget.
  Exceeding the routine envelope latches recovery-required status. The simulator
  does not silently rearm it, move the target to the truth state, or increase its
  authority after a failed case.
- `EventJournal` appends status changes, reacquisitions and issued commands to a
  local outbox. Sequence IDs and acknowledgments support receiver deduplication
  and replay after restart. The test receiver returns acknowledgments when its
  simulated link is available. A runtime storage error stops further appends to the
  potentially damaged file, records the problem and leaves guidance running.
  Capacity overflow and dropped records remain visible in diagnostics.
- The truth plant models both the force and propellant consumption of a stuck
  thruster. Overlapping commanded and uncommanded jets contribute vector forces
  and summed mass flows, split at the exact commanded-burn boundary. Issued Δv,
  actual net propulsive Δv and actual propellant use are separate measurements.

The local outbox is a prototype, not a spacecraft communications stack. The
campaign does not send external notifications or implement authenticated ground
commands, authority handover, durable flight storage or a redundant radio.
Corrupt existing journals fail visibly during initialization; a flight boot
supervisor and redundant storage recovery are still needed. The pending queue
has a capacity limit, but the append-only file also needs a future retention and
rotation policy for long missions.

## Specific remaining failures in this campaign

The stuck-thrust case restores navigation at 19.5 hours in both geometries,
after the injected six-hour thrust fault ends at 18 hours. The large estimated
trajectory error immediately triggers the routine recovery latch. Ending the
fault is part of the test schedule; the controller does not isolate the thruster.
Final trajectory errors are 819.4 km and 316.3 km despite final navigation errors
below 19 m. This is a direct counterexample to treating navigation recovery as
mission recovery.

In P1's force/bias/thrust-loss sequence, routine authority latches at 60 hours.
Navigation later recovers, but the nominal path does not. Its post-event position
RMS is 3.261 km, compared with 1.162 km under matched coasting. The current
corrections therefore worsen this measured outcome. This comparison identifies
an overall failure; it does not isolate a unique faulty command or prove that
one particular estimator parameter caused the excess error.

P1's outage/noisy-return/later-kick sequence exposes a navigation state-machine
gap. The final accepted measurement arrives at 24 hours, near the end of the
noisy return. Subsequent fixes are delivered but rejected. At the first fix
after the 30-hour kick, the accepted track is already more than three hours old,
which blocks the recent-track candidate rule. Continuous packet delivery does
not satisfy the separate observation-gap rule. No candidate is promoted before
run end: navigation finishes 3.756 km wrong and untrusted. This is the code-level
mechanism that prevents the existing fallback from responding to the later
event. A broader loss-of-track recovery policy needs independent validation;
unconditionally reseeding from those same rejected fixes would create a new
sensor-fault vulnerability.

A passive replay reproduced the original metrics, commands, samples and
trajectories exactly. At the first post-kick fix (30.5 h), velocity discrepancy
was 0.182 m/s, velocity NIS was 1609, and position discrepancy was 556 m: all
three candidate thresholds passed. The accepted-track age was 6.5 h, exceeding
the 3 h limit, with cold reacquisition inactive and no candidate present. The
trace is archived in `navigation-lockout-trace.json` alongside the campaign.

## Recommended next failure-handling architecture

Use layered fault detection, isolation and recovery with explicitly bounded
authority. NASA's [fault-detection and response guidance](https://swehb.nasa.gov/spaces/SWEHBVD/pages/102695798/9.07%2BFault%2BDetection%2Band%2BResponse)
supports autonomous protection that preserves resources and commandability and
can complete time-critical actions without waiting for ground. The following
design follows that principle; it has **not** been implemented or validated by
this campaign.

1. **Keep independent health monitors active in every mode.** Navigation,
   propulsion, power, attitude, computing and communications should each retain
   their own health state. A second failure must not be ignored because the
   spacecraft is already in a recovery mode. Use explicit entry, exit and dwell
   conditions to limit repeated switching, and retain evidence for each decision.
2. **Give dangerous hardware faults an independent protective response.** A
   stuck thruster requires a separately controllable shutoff or redundant
   propulsion path. MPC cannot command a failed valve closed through the same
   broken control path. Pressure, valve, acceleration and attitude information
   can constrain an effectiveness estimate; none is provided to this experiment.
   Before another burn, verify actuator authority and update the fuel estimate
   from available onboard measurements rather than assuming every issued burn
   executed as requested.
3. **Rebuild navigation under a defined loss-of-track policy.** Prolonged
   rejection must eventually trigger a separate recovery hypothesis even when
   packets continue arriving. Preserve the old track for comparison, bound
   candidate work, require sustained consistency and, where available, an
   independent observable. Merely widening the acceptance gate or periodically
   resetting to the latest fix would let sensor faults command the spacecraft.
   A single coherent synthetic sensor stream cannot authenticate physical motion.
4. **Separate survival planning from return to the nominal trajectory.** When
   the routine controller's envelope is exceeded, evaluate reachable protective
   trajectories using state uncertainty, degraded thrust authority, fuel reserve
   and body-clearance constraints. Return to the mission corridor only after
   that plan is feasible and validated. Clearing a latch or redefining the
   reference does not demonstrate a safe maneuver.
5. **Continue reporting while retaining ground authority.** Queue evidence,
   uncertainty, health changes, proposed/issued burns and remaining resources.
   Deduplicate and acknowledge reports on reconnection. A future ground-command
   interface needs authentication, epochs, expiry and replay protection; an old
   command must not override a newer emergency response by accident. Ground can
   observe an authorized autonomous mode and intervene when contact permits.

The order of priority is hazardous actuator isolation, dependable navigation
recovery, then a validated survival planner and mission-return optimizer. Lower
tracking RMS cannot compensate for a thruster that cannot be stopped or an
untrustworthy state estimate. Fuel and correction-count optimization belong
inside the feasible, validated operating envelope.

## Evidence boundaries

The reconstructed ARTEMIS data supplies a single initial spacecraft state per
geometry. Later truth trajectories and navigation fixes are simulated. Planetary
ephemerides are prescribed environment inputs. Future spacecraft truth, fault
names, times, directions, gain factors and actual mass do not enter navigation
or control. Exact pre-event comparisons and the unchanged ground-blackout
trajectory check this separation; source review checks the interfaces as well.

The principal score distinguishes restoration of navigation (trusted position
error ≤200 m and velocity error ≤0.02 m/s) from restoration of the nominal
trajectory (≤100 m and ≤0.02 m/s), both sustained for the last six hours after
the fault schedule ends. These errors are truth-based evaluation measurements,
not information available to the onboard policy.

Recorded clearance, propellant reserve and the absence of a modeled collision
are reported separately. Both MPC and coasting can pass those checks in benign
three-day geometries; that does not establish a survival advantage. The
clearance minimum is based on integrator samples and modeled collision events,
not a certified continuous-time lower bound.

The earlier navigation campaign showed empirical uncertainty undercoverage.
Numerically valid covariance is not calibrated confidence. The current 30-minute
sensor/decision interval and multi-fix confirmation delay also do not establish
rapid emergency response. Power, thermal, attitude dynamics, real sensor
observables, radiation effects and permanent hardware damage remain outside the
plant. Only the MPC quadratic-program core runs in integer host emulation; no
FPGA device, synthesis, resource fit, timing or power result is claimed.

## Reproduce

Use an environment with the project's navigation/test dependencies and the
archived ARTEMIS, ephemeris and lunar-gravity files described in the earlier
benchmark documentation. All paths below are local artifacts for this run.

```sh
python scripts/artemis_compound_campaign.py \
  --output /Users/hendrixx./qrc-compound-failures-2026-10-31 \
  --data /Users/hendrixx./qrc-mpc-strategies-2026-10-31/data \
  --kernels /Users/hendrixx./qrc-cislunar-data-2026-10-30 \
  --physics /Users/hendrixx./qrc-force-followup-2026-10-31 \
  --native /Users/hendrixx./qrc-force-followup-2026-10-31/liblunar.so \
  --workers 8
python -m pytest -q
python -m pip wheel . --no-deps --no-build-isolation \
  --wheel-dir /Users/hendrixx./qrc-compound-failures-2026-10-31/wheels
python scripts/report_compound_failures.py \
  --campaign /Users/hendrixx./qrc-compound-failures-2026-10-31 \
  --output docs/artemis-compound-results-2026-10-31
```

The campaign snapshots and hashes numerical sources, physical inputs and its
protocol. It checkpoints every result and retains partial failures. Resuming
requires matching provenance. The report verifies those hashes, scoring,
command budgets, causal prefixes, outbox records, test log and built wheel.
