# ARTEMIS definitive-orbit replay and portable fixed-point PID

This work has two separate tests. The first propagates from one real ARTEMIS
initial state and scores later definitive navigation states. The second tests
a new controller on a simulated spacecraft initialized from that orbit. The
historical spacecraft never executed our commands, so its later trajectory
cannot serve as the truth trajectory of our controlled simulation.

The completed campaign includes 12 physical replays and 85 control runs. The
initial fuel-first tuning failed an October outage test. Selecting a one-hour
response from the existing September training results, then evaluating on fresh
November initial conditions and noise seeds, gave 19.6–28.9 m steady tracking
RMS with 24-bit arithmetic. This is a conditional simulation result, not measured
flight-navigation accuracy or a demonstrated shape-coordinate advantage.

The [machine-readable report](artemis-fpga-pid-results-2026-10-31.json),
[initial campaign plot](artemis-fpga-pid-results-2026-10-31.png),
[fresh comparison plot](artemis-fpga-pid-refinement-2026-10-31.png) and
[final register configuration](artemis-fpga-pid-config-2026-10-31.json) retain
the failed cases, revised choice, arithmetic checks and provenance.

## Data pulled and checked

The local campaign at `/Users/hendrixx./qrc-artemis-fpga-2026-10-31/` contains
both public `ARTEMIS_P1_DEF_2009022_2012022.bsp` and
`ARTEMIS_P2_DEF_2009022_2012022.bsp`, plus 32 definitive V02 CDF files for
February 1–4, September 1–4, October 1–4 and November 1–4, 2011. November's
eight files were acquired for the fresh evaluation. Download manifests record
URLs, byte counts and SHA-256 hashes. The existing P1 cache was reused and
verified; P2 and the CDF files were downloaded.

The [mission file specification](https://spdf.gsfc.nasa.gov/pub/data/themis/documents/tdas/docs/thm_soc_124_L1_File_Def.pdf)
and the downloaded CDF global attributes distinguish the products:

- V00 and V01 use predicted spacecraft ephemerides. V01 adds telemetry-derived
  spin information; that does not make its orbit an observed trajectory.
- V02 adds definitive position from orbit determination. V03 adds attitude
  improvements. These are reconstructed products, not raw tracking samples.
- The older `artemis_live/archive` is mostly V02 (1,456 files), with eight
  V01 files. Its calibration must not be dismissed as entirely prediction-based.
  The recent live directory contains V01 predictions and requires separate
  labeling when scored.

The [NAIF product readme](https://naif.jpl.nasa.gov/pub/naif/THEMIS/kernels/spk/aareadme.txt)
identifies the DEF products as regular orbit-determination solutions supplied
by mission navigation. NAIF checked their structural integrity but describes
them as unofficial and does not certify their ephemeris validity.

SPK queries use spacecraft IDs -651/-652, geometric J2000 states, Moon origin,
SI units and elapsed TDB seconds. Actual segment descriptors say frame 1 and
SPK type 12. For a cross-check, the CDF's documented true-equator/equinox-of-date
positions are rotated with IAU1976/1980 precession/nutation. SPK/CDF position
differences are about 9–11 m RMS over these arcs. They can share upstream orbit
solutions, so this agreement is not independent sensor validation.

## Initial-only propagation

The force model contains degree-60 GRGM900C lunar harmonics, DE440 planetary
ephemerides, Earth/Sun differential gravity, other planetary tides, Earth J2,
first-PN corrections and solar radiation pressure with eclipse handling.
Cr*A/m = 0.016 m²/kg is a declared nominal input. September sensitivity runs
also use 0 and 0.032; no later spacecraft states are fitted into the force.
Lunar elastic-tide conventions and actual spacecraft attitude/area/mass history
remain unresolved inputs.

| Arc starting at 00:00 UTC | 24 h position error | 48 h error | 72 h error | Supplied maneuver flag |
| --- | ---: | ---: | ---: | --- |
| P1, 2011-02-01 | 41.558 km | 223.522 km | 346.970 km | First nonzero code at 18.57 h |
| P2, 2011-02-01 | 18.922 km | 9.854 km | 15.634 km | First nonzero code at 2.15 h |
| P1, 2011-09-01 | 8.234 km | 3.324 km | 2.697 km | None in the sampled arc |
| P2, 2011-09-01 | 0.795 km | 685.441 km | 7,837.852 km | First nonzero code at 42.02 h |
| P1, 2011-10-01 | 0.075 km | 0.073 km | 2.013 km | None in the sampled arc |
| P2, 2011-10-01 | 515.387 km | 9,212.939 km | 9,163.625 km | First nonzero code at 19.40 h |
| P1, 2011-11-01 | 1.313 km | 0.345 km | 0.439 km | None in the sampled arc |
| P2, 2011-11-01 | 1.021 km | 2.130 km | 3.245 km | None in the sampled arc |

These full-arc results are retained. Without historical maneuver inputs, a coast
model is not expected to reproduce the portions after a burn. A nonzero supplied
status is evidence of an event; it does not supply the actual thrust vector,
duration or magnitude. Conversely, zero flags do not certify an uninterrupted
coast or a continuous reference product.

Before the first flagged event, the P1 February replay has 21.1 m RMS over
18.5 h, P2 February 0.25 m RMS over 2 h, and P2 October 13.7 m RMS over
19.33 h. Those are different durations, not directly comparable accuracy
benchmarks. P1 October reaches 75.5 m at one day and 73.0 m at two days.
Reconstructed starting states may incorporate later tracking, so even these
results are retrospective propagation replays rather than blind forecasts.

The two apparently unflagged P1 arcs also fail a reference self-consistency
check near the sharp residual rises: one-second position increments disagree
with the trapezoidal integral of the supplied velocity by approximately 428 m
(September) and 618 m (October), compared with a sub-micrometre median away
from those transitions. The narrowed-interval diagnostic is saved separately.
This is evidence against treating every position residual as missing force.
The diagnostic was performed after inspecting the errors and is not a
preregistered event detector. No inferred impulse or reference correction is
fed back into the forecasts.
Bisecting the intervals leaves the September/October position defects at
427.962/618.046 m over only 15.26 microseconds, with velocity changes of
0.00339/0.00030 m/s. Ordinary omitted forces cannot cause those position jumps
without the corresponding velocity evolution. This identifies a product
continuity problem at those instants; it does not explain every residual.

## PID and FPGA interface

Hardware has not been selected. `hardware/orbit_pid_core.cpp` is a portable
integer-only HLS candidate. Its interface accepts tracking errors and a
model-derived feedforward impulse; it does not accept an entire ephemeris or
run an optimizer. In this restricted Jacobi implementation the actual and
reference states receive the same time-dependent translation, so their position
and velocity errors equal the corresponding Moon-relative errors.

The control law uses proportional position error, filtered velocity error,
an integral contribution and model-based gravity feedforward. It has conditional
integration under output saturation, an integral limit, symmetric axis limits
whose vector norm fits the actuator budget, and explicit invalid/stale-input
inhibition. Arithmetic range faults also inhibit output and preserve state.
Derivative input comes from estimated velocity, avoiding differencing noisy
position fixes. The caller must enforce the configured sample interval and
provide current navigation/command information.

The tested arithmetic formats use 16/8, 24/16 and 32/24 total/fractional bits,
with a signed normalized range approximately [-128,128). Position error is
scaled by 4096 m; velocity error and command delta-v by 0.25 m/s. Absolute
barycentric coordinates are not forced into this small error format. Each
product uses a signed 64-bit intermediate, rounds to nearest with ties away
from zero, and explicitly detects range overflow. The three axes have fixed
loop bounds; there is no allocation, convergence loop or transcendental
function in the PID core.

`fpga_pid.py` includes an independent Python integer oracle and the C ABI
bridge. Tests compare 1,500 sequential randomized calls at each width, test
anti-windup, stale-data behavior, rounding and atomic range-fault handling.
The C++ testbench also passes address/undefined-behavior sanitizers. These
establish host arithmetic behavior, not RTL timing or device qualification.
The portable C ABI stores values in int32 buffers even for narrower formats;
arithmetic precision comparisons alone do not establish FPGA resource savings.

`run_orbit_pid_hls.tcl` requires `FPGA_PART` and accepts `FPGA_CLOCK_NS`. It
requests C simulation, synthesis and RTL co-simulation. The requested clock is
a target constraint. No Vitis/Vivado HLS or RTL simulator was found in this
environment; no resource, achieved-frequency, latency or power result is claimed.
AMD's [fixed-point documentation](https://docs.amd.com/r/2023.1-English/ug1399-vitis-hls/Fixed-Point-Identifier-Summary)
explains why rounding/saturation assignments must be explicit and why their
hardware cost cannot be inferred from a numerical width alone.

The full navigation filter, high-degree force evaluation and trajectory
propagation currently execute in the host harness. The new core covers PID
feedback, not an entire FPGA-resident orbit-determination and emergency-planning
system. Actual ARTEMIS spin-constrained thruster allocation is also not modeled:
the controlled vehicle retains the explicitly assumed 500 kg, 0.2 N equivalent
omnidirectional actuator and 220 s Isp. It uses ARTEMIS orbit geometry, not a
claim that the historical spacecraft could execute these exact commands.

## Initial fuel-first tuning and its failed evaluation

Training uses P1/P2 September initial conditions and two noise/fault seeds per
probe. Nine candidates span 1/3/6 h response times and integral strengths
0/0.1/0.3, with 30-minute updates. The pilot uses a separate seed. Each
simulation generates a planned nominal trajectory from the initial state only,
then propagates a separate plant with declared acceleration bias, a force
episode and imperfect burns. Navigation fixes are synthetic 30 m/0.002 m/s
per axis, delivered 60 seconds late. Fault cases add a six-hour outage and one
corrupt fix. These measurements are not claimed to be actual onboard telemetry.

A candidate must keep every training scenario's steady RMS below 1 km and
sampled maximum below 10 km, without collision; qualifying candidates are
ranked by mean commanded delta-v. Steady scoring excludes the first 12 hours;
full metrics remain available. Precision is then checked on training cases
using the chosen gains. October cases and new seeds provide the frozen
controller evaluation. The physical replay table above remains separate.

All 36 gain-training runs completed. The minimum-fuel qualifying configuration
was a six-hour response time and zero integral strength. This initial
operating setting was **PD plus model feedforward**; the implementation still
supports the tested integral path and its anti-windup logic. Its worst training
steady RMS was 129.07 m and mean three-day delta-v was 0.36282 m/s.
The one-hour response with integral disabled gave a lower worst RMS of 40.74 m
at 0.64956 m/s mean delta-v. Those represent an accuracy/fuel tradeoff rather
than a universally optimal tuning.

The three-hour settings failed on the P2 outage case: the estimator rejected
returning fixes and the controller inhibited routine corrections after the
estimate became stale. Worst steady RMS values ranged from 2.01 to 13.20 km.
Both one-hour and six-hour settings passed the training requirements. This
non-monotonic outcome involves the orbit, sampled controller and estimator;
it is not evidence that a response-time parameter alone determines stability.

For the selected six-hour configuration, 24-bit arithmetic with 16 fractional
bits passed all four training precision comparisons. Its largest relative
steady-RMS change was 0.223% and largest delta-v change 0.182%. The 32/24 format
also passed (0.0053%/0.00067%); the 16/8 format failed (up to 494%/135%).
The selection rule required absolute relative changes no larger than 1% for
both metrics, at most 10 m change in sampled maximum error, and no collision or
arithmetic range fault. This selects arithmetic precision for the prototype;
the current int32 C ABI and missing synthesis prevent a resource-savings claim.

The subsequent October evaluation failed the intended accuracy criterion:
P1's outage case reached 6,246 m steady RMS in floating point (6,242 m at
24 bits), against the 1 km criterion. Ordinary P1/P2 and outage P2 gave
415/223/688 m in floating point. The failure survives increased precision;
it is a controller/estimator performance issue, not a fixed-point explanation.

## Accuracy-first repair and fresh November evaluation

After the October failure, the selection objective changed to minimum worst-case
training RMS among the same feasible candidates. Only the already-computed
September training table selected the revised gain: **one-hour response, zero
integral strength**. The new protocol froze that choice and fresh November dates
and seeds before running them. October is no longer a fresh test for this
revised design. No November outcomes selected gains or precision.

For this revised gain, 24/16 and 32/24 formats were rechecked on all four
September training scenarios. The 24-bit format passed, with maximum relative
RMS/delta-v changes of 0.0621%/0.0353%. The final register configuration uses
24 total signed bits with 16 fractional bits, a 1,800 s update interval,
Kp = 1/(3600 s)², Kd = 2/(3600 s), Ki = 0, and a 1,800 s velocity-filter
time constant. Recompute discrete coefficients if the update interval, scales
or actuator interpretation change.

| Fresh November scenario | Original six-hour RMS | Revised float RMS | Revised 24-bit RMS | Original / revised 24-bit delta-v |
| --- | ---: | ---: | ---: | ---: |
| P1, ordinary | 1,413.72 m | 26.39 m | 26.38 m | 2.4158 / 1.0956 m/s |
| P1, outage and bad fix | 972.89 m | 19.65 m | 19.64 m | 1.6876 / 0.8283 m/s |
| P2, ordinary | 971.00 m | 22.66 m | 22.66 m | 1.2942 / 0.6964 m/s |
| P2, outage and bad fix | 462.65 m | 28.85 m | 28.88 m | 0.6648 / 0.7700 m/s |

Each simulation lasts three days; steady RMS excludes the first 12 hours.
Full-run maximum sampled error is about 515 m, including the initial offset.
The fault cases reject the injected bad fix and inhibit 11 commands while fixes
are stale. Navigation is considered stale after more than one hour without a
fresh fix. No arithmetic range fault or simulated collision occurred in these
eight revised evaluation runs. Routine burns still occur every available
30-minute epoch; this does not demonstrate sparse corrections planned months
ahead or a validated emergency maneuver supervisor.

On exactly matched November cases and random draws, revised floating point
reduces mean steady RMS from 955.06 m to 24.39 m (**97.45%**) and mean commanded
delta-v from 1.5156 to 0.8478 m/s (**44.07%**). Three cases use less delta-v;
P2's outage case uses more. The largest difference between revised 24-bit and
floating-point steady RMS is 0.036 m. Holding plant, initial conditions and
random inputs fixed supports attributing these simulated differences to the
gain change. Four short scenarios, a generic actuator and synthetic direct-state
fixes leave mission-wide robustness and flight fuel savings unestablished.

The source snapshot records the exact numerical implementation used. After the
runs, a docstring was corrected: Cr*A/m was a nominal input with sensitivity
checks, not a parameter trained by this campaign. The older ARTEMIS archive is
mostly V02, so an earlier script's phrase "V01-based calibration" is not a valid
description of that archive. A verification receipt records that removing
docstrings makes the frozen and current numerical module ASTs identical.

## Reproduction and checks

Use the existing DE440/GM/leap-second kernels and the gravity/orientation/native
library described in the earlier force-model guide. From the repository:

```sh
python -m pip install -e '.[navigation,test]' matplotlib
python -m qrc_pulsar_bench.artemis_download /tmp/artemis-pid

python -m qrc_pulsar_bench.artemis_replay \
  --data /tmp/artemis-pid/data --kernels /path/to/planetary-kernels \
  --physics /path/to/lunar-inputs --native /path/to/liblunar.so \
  --output /tmp/artemis-pid/replay --case P1_oct --cram 0.016

PYTHONPATH=. python scripts/artemis_pid_campaign.py train \
  --output /tmp/artemis-pid --kernels /path/to/planetary-kernels \
  --physics /path/to/lunar-inputs --native /path/to/liblunar.so
# Repeat the campaign command with phase 'precision', then freeze that selection:
python scripts/report_artemis_pid.py /tmp/artemis-pid --select-precision
# Repeat the campaign command with phase 'evaluate'.

python scripts/prepare_artemis_pid_refinement.py /tmp/artemis-pid
PYTHONPATH=. python scripts/artemis_pid_accuracy_validation.py precision \
  --output /tmp/artemis-pid --kernels /path/to/planetary-kernels \
  --physics /path/to/lunar-inputs --native /path/to/liblunar.so
# Repeat the accuracy-validation command with phase 'fresh'.
PYTHONPATH=. python scripts/artemis_pid_compare.py \
  --output /tmp/artemis-pid --kernels /path/to/planetary-kernels \
  --physics /path/to/lunar-inputs --native /path/to/liblunar.so

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
```

The full report additionally uses the replay/event/continuity audit receipts;
the executed audit scripts and source snapshots are in the local campaign
directory. The public runner checks source, protocol and physical input hashes
between phases and rebuilds the host PID library from the checked source.
Use a new output directory for a campaign on edited source; exact replay of the
recorded source hashes requires the saved source snapshot. Run these scripts
from the repository: the host compiler bridge expects its `hardware/` directory.
The downloader refuses to substitute predictive files for missing definitive
V02 products. A data-flow test changes later reference samples and verifies that
the propagated prediction is unchanged while its score changes.

The complete test suite passed **334 tests**, including 14 new replay/PID tests,
with one existing PINT extended-precision warning on this platform. HLS/device
verification remains a separate required step after selecting a device/toolchain.

The PID core itself is not a predictive emergency supervisor and does not
certify collision avoidance or global nonoscillation. A stable local linear
feedback test and successful finite scenarios cannot establish those claims.
