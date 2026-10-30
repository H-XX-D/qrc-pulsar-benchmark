"""
=====================================================================================
CLI Execution Verification Tests
=====================================================================================
Ensures that all CLI commands execute cleanly without NameError or missing imports.
=====================================================================================
"""

import pytest
import os
import subprocess
import sys

def run_cli(*args):
    cmd = [sys.executable, "-m", "qrc_pulsar_bench.cli"] + list(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def test_cli_list():
    res = run_cli("list")
    assert res.returncode == 0
    assert "PSR_J0737-3039" in res.stdout

def test_cli_run():
    res = run_cli("run", "--system", "PSR_J0737-3039")
    assert res.returncode == 0
    assert "Periastron Advance" in res.stdout

def test_cli_all():
    res = run_cli("all")
    assert res.returncode == 0
    assert "MASTER BENCHMARK ACROSS ALL 5 REFERENCE SYSTEMS" in res.stdout

def test_cli_simulate():
    res = run_cli("simulate", "--system", "PSR_J0337+1715", "--orbits", "2", "--points", "100")
    assert res.returncode == 0
    assert "TIME-EVOLUTION SIMULATION ON SHAPE SPHERE S²" in res.stdout
    assert "Energy Conservation Error" in res.stdout
    assert "Shape Sphere Invariance Error" in res.stdout

def test_cli_compare():
    res = run_cli("compare", "--orbits", "2")
    assert res.returncode == 0
    assert "LIVE MEASURED BENCHMARK" in res.stdout
    assert "Relational Jacobi Solver Wall-Time" in res.stdout
    assert "Cartesian 18-ODE Solver Wall-Time" in res.stdout

def test_cli_custom(tmp_path):
    sample_file = tmp_path / "test.csv"
    with open(sample_file, "w") as f:
        f.write("# MJD, res, unc\n55000.0, 0.1, 1.0\n55010.0, -0.1, 1.0\n55020.0, 0.05, 1.0\n")
    res = run_cli("custom", "--path", str(sample_file))
    assert res.returncode == 0
    assert "Total Observations Loaded" in res.stdout

def test_cli_zenodo():
    paths = ["psr_j0337_zenodo_residuals.csv", "../psr_j0337_zenodo_residuals.csv"]
    csv_path = next((p for p in paths if os.path.exists(p)), None)
    if not csv_path:
        pytest.skip("Zenodo dataset (psr_j0337_zenodo_residuals.csv) not found locally")

    res = run_cli("zenodo", "--path", csv_path)
    assert res.returncode == 0
    assert "Total Radio Observations" in res.stdout
