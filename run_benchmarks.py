#!/usr/bin/env python3
"""
Standalone Benchmark Runner for QRC-PULSAR-BENCH
Executes all benchmarks without needing pip installation.
"""

import sys
import os

# Ensure local package is in Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from qrc_pulsar_bench.cli import main

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Default action: run all benchmarks
        sys.argv.append("all")
    main()
