#!/usr/bin/env bash
# Run the three mathematical checks and report each exit status.
# Exit status is 0 only if all three scripts exit with 0.
set -u
cd "$(dirname "$0")"
status=0
for script in check_gate1_selection.py check_gate_results.py check_gate5_family.py; do
  if python3 -I -B "$script" > /dev/null; then
    echo "$script: all observed outcomes match expectations"
  else
    echo "$script: MISMATCH or error (run it directly to see the output)"
    status=1
  fi
done
exit "$status"
