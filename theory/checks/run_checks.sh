#!/usr/bin/env bash
# Run all mathematical check suites across Gates 0 through 6 and report each exit status.
# Exit status is 0 only if all scripts exit with 0.
set -u
cd "$(dirname "$0")"
status=0
for script in \
    check_gate0_premetric.py \
    check_gate1_selection.py \
    check_gate_results.py \
    check_gate2_cone_algebra.py \
    check_gate3_quantum_born.py \
    check_gate4_gravity_dewitt.py \
    check_gate5_family.py \
    check_gate6_mass_channels.py \
    check_all_gates_empirical_confrontation.py \
    check_all_gates_truthful_equations.py \
    check_qrc_grand_synthesis_first_principles.py \
    check_qrc_truthful_lagrangian.py; do
  if python3 -I -B "$script" > /dev/null; then
    echo "$script: all observed outcomes match expectations"
  else
    echo "$script: MISMATCH or error (run it directly to see the output)"
    status=1
  fi
done
exit "$status"
