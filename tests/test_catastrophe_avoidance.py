"""Unit tests for low-power orbital mechanics guidance and catastrophe avoidance system."""

import pytest
import numpy as np

from qrc_pulsar_bench.catastrophe_avoidance import (
    CatastropheAvoidanceFlightSystem,
    PowerBudgetComparison,
    CatastropheHazardState,
)


def test_power_budget_reduction():
    """Verifies that the micro-FPGA guidance architecture achieves >90% power and mass reductions."""
    flight_sys = CatastropheAvoidanceFlightSystem(target_system="Jupiter-Callisto")
    conv_w, edge_w, conv_kg, edge_kg = flight_sys.compute_total_power_and_mass()

    # Total conventional power is typically 130-140 W
    assert conv_w >= 120.0
    # Low-power edge guidance operates at <5 W total
    assert edge_w <= 4.0
    # More than 95% power reduction
    power_reduction = (conv_w - edge_w) / conv_w * 100.0
    assert power_reduction > 95.0

    # Avionics mass drops from ~28 kg to <1 kg
    assert conv_kg > 20.0
    assert edge_kg < 1.0
    mass_reduction = (conv_kg - edge_kg) / conv_kg * 100.0
    assert mass_reduction > 95.0


def test_jacobi_energy_confinement_evaluation():
    """Verifies Jacobi integral computation and Hill's zero-velocity surface confinement."""
    flight_sys = CatastropheAvoidanceFlightSystem(target_system="Jupiter-Callisto")

    # Safe state: position near Callisto L1 with low relative velocity (high Jacobi energy C_J > C_L1)
    r_safe = np.array([1.0 - 0.0264 + 0.005, 0.0, 0.0])
    v_safe = np.array([0.0, 0.001, 0.0])
    c_j_safe = flight_sys.compute_jacobi_integral(r_safe, v_safe)
    assert c_j_safe > 3.0018  # Above C_L1 gateway

    hazard_safe = flight_sys.assess_hazard_and_catastrophe_risk(
        r_norm=r_safe,
        v_norm=v_safe,
        available_thrust_n=2.0,
        spacecraft_mass_kg=15.0,
        imu_valid=True,
        optical_flow_valid=True,
        range_to_surface_km=500.0,
        closing_velocity_mps=10.0,
    )
    assert hazard_safe.is_energetically_confined is True
    assert hazard_safe.hazard_score < 0.30
    assert hazard_safe.recommended_abort_action is None


def test_catastrophe_avoidance_reflex_trigger():
    """Verifies that an uncontained, high-speed terrain plunge triggers an immediate abort."""
    flight_sys = CatastropheAvoidanceFlightSystem(target_system="Jupiter-Callisto")

    # Hazardous state: high velocity, low Jacobi energy, closing fast on terrain, IMU failed
    r_peril = np.array([0.98, 0.01, 0.0])
    v_peril = np.array([0.15, -0.10, 0.0])  # Excessive relative velocity
    
    hazard_peril = flight_sys.assess_hazard_and_catastrophe_risk(
        r_norm=r_peril,
        v_norm=v_peril,
        available_thrust_n=0.5,
        spacecraft_mass_kg=15.0,
        imu_valid=False,  # Sensor blindness
        optical_flow_valid=True,
        range_to_surface_km=25.0,
        closing_velocity_mps=350.0,  # Time to impact < 100 s
    )
    assert hazard_peril.is_energetically_confined is False
    assert hazard_peril.hazard_score >= 0.70
    assert len(hazard_peril.active_threats) >= 2
    assert "EMERGENCY_RETROGRADE_ABORT_BURN" in hazard_peril.recommended_abort_action


def test_dsn_reaction_comparison():
    """Verifies the latency advantage of onboard autonomous reflex vs Earth ground intervention."""
    flight_sys = CatastropheAvoidanceFlightSystem()
    metrics_jupiter = flight_sys.calculate_dsn_reaction_comparison(distance_au=5.20)

    # At 5.2 AU, round-trip light time is ~86 minutes
    assert metrics_jupiter["round_trip_light_time_min"] > 80.0
    # Autonomous reflex responds in 35 milliseconds
    assert metrics_jupiter["onboard_autonomous_reflex_ms"] == 35.0
    # Speedup factor is >100,000x faster than human ground controllers
    assert metrics_jupiter["autonomy_speedup_factor"] > 100000.0


def test_autonomy_inversion_audit():
    """Verifies that human-in-the-loop is viable in LEO but dead beyond the Moon."""
    flight_sys = CatastropheAvoidanceFlightSystem()
    audit = flight_sys.generate_fleet_autonomy_inversion_audit()

    # LEO: Ground viable, inversion ratio < 0.01
    leo = audit[0]
    assert leo.human_in_loop_viable is True
    assert leo.inversion_ratio < 0.05
    assert leo.survival_criticality_rank > 1

    # Ceres, Jupiter, Saturn, Uranus, Neptune: Inversion occurs, human-in-the-loop NOT viable
    for profile in audit[2:]:  # From Ceres outward
        assert profile.human_in_loop_viable is False
        assert profile.inversion_ratio > 100.0
        assert profile.survival_criticality_rank == 1
        assert profile.onboard_autonomy_criticality_weight == 1.00

    # At Neptune: Inversion ratio > 10,000x
    neptune = audit[-1]
    assert neptune.inversion_ratio > 10000.0
    assert neptune.round_trip_light_time_s > 29000.0

