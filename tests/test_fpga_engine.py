"""Tests for the FPGA Orbit Engine & Pipelined Hardware Datapath."""

import pytest
import numpy as np
from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.constants import AU
from qrc_pulsar_bench.fpga_orbit_engine import (
    FPGASpecification,
    PipelinedInvCubeUnit,
    FPGAOrbitEngine,
    HardwareCycleReport,
)


def test_pipelined_inv_cube_unit_accuracy():
    """Verify that LUT + 2-stage Newton-Raphson InvCubeUnit matches exact fp64 r^-3."""
    unit = PipelinedInvCubeUnit(lut_bits=10)
    # Test across 6 decades of physical distances (1,000 km to 10 AU)
    radii = np.logspace(6.0, 12.0, 50)
    r_sq = radii**2

    exact_inv_r3 = 1.0 / (radii**3)
    hw_inv_r3 = unit.evaluate(r_sq)

    rel_error = np.abs((hw_inv_r3 - exact_inv_r3) / exact_inv_r3)
    max_rel_error = float(np.max(rel_error))

    # 2 Newton-Raphson iterations deliver near-double precision (< 5e-11)
    assert max_rel_error < 5e-11, f"InvCubeUnit max relative error too high: {max_rel_error:.2e}"


def test_fpga_hardware_timing_and_throughput():
    """Verify hardware cycle counts and deterministic latency on Xilinx UltraScale."""
    spec = FPGASpecification(clock_frequency_mhz=150.0)
    engine = FPGAOrbitEngine(PSR_J0337_1715, fpga_spec=spec, order=4)

    latency_cycles, ii, latency_ns = engine.get_hardware_timing()

    # 3 sub-stages of ~31 cycles + 4 cycle Roemer stage = ~97 cycles
    assert 80 <= latency_cycles <= 120
    assert ii == 1  # Pipelined throughput: 1 step per clock
    # At 150 MHz (6.67 ns period), latency should be ~600-800 ns
    assert 500.0 <= latency_ns <= 1000.0


def test_fpga_orbit_simulation_energy_and_dsp_budget():
    """Verify 10-day hardware simulation: bounded energy error and <5% DSP utilization."""
    engine = FPGAOrbitEngine(PSR_J0337_1715)
    times, pos, roemer, report = engine.run_hardware_simulation(
        duration_days=5.0,
        dt_s=600.0
    )

    assert len(times) == 720
    assert pos.shape == (720, 3, 3)
    assert len(roemer) == 720

    # Symplectic energy conservation on chip (< 1e-6)
    assert report.energy_error < 1e-6
    # DSP budget check: uses ~72 DSPs (< 3% of XQRKU060's 2,760 DSPs)
    assert report.estimated_dsp_slices < 100
    assert report.dsp_utilization_pct < 5.0
    assert report.hardware_validated is False
    assert report.position_error_vs_fp64_m is None
    # Roemer delays should be in the tens-of-seconds range for J0337
    assert np.max(np.abs(roemer)) > 1.0
