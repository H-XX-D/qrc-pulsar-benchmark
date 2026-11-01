# MPC input hardening and extended mission tests

The development starting point remains the sparse MPC policy from the previous
strategy search. Hardening adds checks around it without treating a new estimator
or a longer prediction through an outage as an automatic improvement.

## Implemented safeguards

- Both QP paths validate dimensions, finite values, scaling, acceleration and
  stopping parameters, symmetry and positive definiteness. The integer bridge
  checks convexity again after quantization. A failed floating-point solve now
  returns a zero command vector, matching the native failure behavior.
- The native library exports its ABI version, word width, fractional width,
  workspace and iteration budget. The Python bridge rejects a mismatched or
  unversioned library. Rebuild old `.so` files with `compile_core`; interpreting
  a 24-bit library as a 32-bit library is no longer silently accepted.
- Controller entry checks reject nonfinite state, malformed covariance, future
  fixes, estimates at the wrong decision epoch, and duplicate/reversed command
  epochs. Faults clear queued warm starts and pending schedules.
- Covariance validation uses correlation units. Tiny negative acceleration
  variances can no longer pass a tolerance intended for position variances.
- Replayed observations are rejected before they can shrink filter covariance
  a second time. Delayed forecasts remain separate from committed filter state.
- Force-linearization and bias-transition caches are bounded by the configured
  horizon (12 entries each for this four-step MPC). Recent onboard diagnostics
  are capped at 256 entries. The extended simulator prunes the incorporated burn
  ledger and retains only the latest truth rollout needed by the delayed sensor.

The host still supplies the native QP core with a checked convex problem. These
checks do not implement radiation fault tolerance or integrity checking on FPGA
memory and buses. No FPGA board has been selected or synthesized.

## Experimental outage extensions

`FilterSettings.bias_correlation_s` adds a first-order decaying acceleration bias.
Zero preserves the constant-bias model. Mean propagation and covariance use the
same decay rate, and a unit test checks both against an analytic solution.

`GuardedMPC` can act on a propagated nine-state estimate after fixes become stale,
provided age and covariance remain within configured bounds. The default guard
allows at most eight hours since the last fix, position sigma 200 m and velocity
sigma 0.03 m/s. After one hour, requested delta-v is capped at 0.03 m/s. It also
inflates the nonlinear first-burn body-clearance floors by three times the
current position sigma. An explicit invalid-health flag always inhibits output.
Neither this covariance allowance nor a first-burn rollout establishes a global
safety or collision-probability guarantee.

The four-profile development test compares the original policy, decay alone,
guarded control alone, and their combination. All use the same force law,
starting states, noise draws and finite-burn errors. The January 10 geometries
and seeds are reused development evidence, not new holdout data.

**The original sparse policy won the declared development rule.** Guarded
operation improved the old worst case, but increased mean error, fuel and burn
count across the four cases. The two-hour bias-decay settings failed broader
acceptance limits in some cases. These extensions remain optional experiments;
neither is made the selected default merely because it is more elaborate.

## Extended validation

The first frozen campaign uses new January 15/18 initial geometries and seeds for
three- and seven-day cases. They include six-, eight- and twelve-hour outages,
disturbances that stop or persist through outages, larger bias and doubled
thrust-error standard deviation. It exposed one seven-day loss of tracking with
more than 20 m/s of routine correction demand. The proposed endurance phase was
then replaced by a supervised revision; the failed data remains unchanged.

`RoutineMPC` adds a causal estimated-error envelope (5 km position / 0.5 m/s
velocity) and a rolling budget of 1 m/s of issued delta-v per 24 hours. Commands
that exceed the budget are rejected whole, preserving the nonlinear screen's
meaning. Exceeding the tracking envelope latches `recovery_required`; it does
not resume merely because fixes return. Explicit rearming requires fresh,
timestamped navigation inside the envelope and does not refund spent budget.
The controller also enforces fresh-fix age and decision cadence internally.
These generic simulation limits require mission-specific validation.

The supervised revision replays a benign and failed case and then uses new
January 20 initial geometries/seeds for separate P1/P2 30-day endurance tests,
with recurring six-hour outages, one twelve-hour outage, and alternating
transient/persistent disturbances. The replay reduces runaway fuel demand but
does not recover the lost trajectory. A separately validated recovery planner
remains necessary; zero command output does not establish safety.

The failed seven-day P1 reference also comes within 37.8 km of the Moon,
below the controller's 50 km clearance floor. It is not an admissible reference
for that operating constraint. Returning fixes are repeatedly rejected before
navigation recovers and the routine supervisor latches at day 2.5625. The
recorded sequence identifies reference and reacquisition problems; it does
not establish that either alone caused the loss of tracking. Future recovery
work must address both before treating this as an autonomous mission system.

The spacecraft archive loader reads only the initial state. Everything later,
including sensor fixes, is simulated; planetary ephemerides remain prescribed
force inputs. Reconstructed initial states can incorporate later historical
tracking and therefore do not constitute a real-time initial-navigation test.

Results, including failures, are in the [executed report](artemis-hardening-results-2026-10-31.md).
The completed campaign has 44 runs totaling 280 simulated days, including
reused development cases and arithmetic comparisons. The new 30-day results are:

| Initial geometry | Steady RMS | Steady peak | Total delta-v | Burns | Outcome |
|---|---:|---:|---:|---:|---|
| P1, January 20 | 42.416 m | 369.293 m | 2.18633 m/s | 866 | Tracking limits met; two clearance rejections |
| P2, January 20 | 36.167 m | 247.696 m | 2.26994 m/s | 905 | Tighter tracking limits met; no solver/screen rejection |

These are float64 results. The 32/24 core gives identical burn counts, at most
0.000339% RMS difference and 0.00439% delta-v difference. P1's nominal path
reaches 45.9 km clearance and violates the 50 km operating floor; its tracking
success does not establish reference feasibility. Neither 30-day case required
the recovery latch or budget wait. The latch is exercised by the failed-case
replay; budget exhaustion is exercised by the unit tests.

Five-minute scoring excludes the first 12 hours for steady metrics and also
records full-arc errors. A 30-day simulation continually replans a two-hour MPC
horizon; it does not solve one globally optimal month-long transfer.

## Reproduction and artifacts

The first campaign is `/Users/hendrixx./qrc-mpc-hardening-2026-10-31/`;
the supervised revision is `/Users/hendrixx./qrc-mpc-supervised-2026-10-31/`.
They store frozen protocols, numerical source copies, input hashes, complete
time histories, arithmetic checks, selected settings and verification receipts.

```sh
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python scripts/artemis_resilience_campaign.py develop \
  --output /tmp/mpc-hardening --data /path/to/ARTEMIS-SPKs \
  --kernels /path/to/planetary-kernels --physics /path/to/lunar-inputs \
  --native /path/to/liblunar.so --workers 4
# Repeat with precision and fresh, using the same arguments.
# The supervised revision has regression and endurance phases:
PYTHONPATH=. python scripts/artemis_supervised_campaign.py regression \
  --output /tmp/mpc-supervised --previous /tmp/mpc-hardening \
  --data /path/to/ARTEMIS-SPKs --kernels /path/to/planetary-kernels \
  --physics /path/to/lunar-inputs --native /path/to/liblunar.so --workers 4
# Repeat that command with endurance.
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q
python scripts/report_mpc_hardening.py /tmp/mpc-hardening /tmp/mpc-supervised
```

The report verifies the archived runner/source hashes, full test logs, built
wheel modules, and native sanitizer/configuration-check receipts from the
executed campaign directories.
The listed simulation commands generate new result sets. The final audit also
requires the separate test/build/native-check receipts retained in the executed
directories. Current runners automatically archive themselves and reject
resuming with a changed runner; the original executed runner copies remain
in each campaign's `source/scripts` directory. Later runner changes fix selected
profile pairing and bind the supervised input arguments to the actual hashed
files. They do not alter the archived numerical results.
Numerical-source changes require a new output directory. The previous strategy
campaign remains reproducible using its own frozen source copies.

The HLS script now defaults to 32 total / 24 fractional bits and checks format
overrides. It still requires an explicitly supplied `FPGA_PART`. Host compilation,
integer-oracle agreement and sanitizer checks are distinct from HLS synthesis,
RTL timing, resource use, power, or flight qualification.

The deterministic native fault/range test is retained as
`hardware/orbit_mpc_fault_testbench.cpp`. It sweeps problem dimensions and
extreme gradients, checking that failures zero every output and successful
outputs remain in bounds. Reproduce its 600 cases per arithmetic format with:

```sh
clang++ -std=c++17 -fsanitize=address,undefined -fno-omit-frame-pointer \
  -DQRC_MPC_BITS=32 -DQRC_MPC_FRAC=24 -Ihardware \
  hardware/orbit_mpc_core.cpp hardware/orbit_mpc_fault_testbench.cpp \
  -o /tmp/mpc-fault32
/tmp/mpc-fault32
# Repeat with 24 total / 16 fractional bits and a distinct output name.
```
