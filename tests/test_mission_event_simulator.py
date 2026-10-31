"""Unit tests for the Mission Event Injection and Anomaly Response Simulator."""

import pytest
import numpy as np

from qrc_pulsar_bench.mission_event_simulator import (
    MissionEventSimulator,
    InjectedEvent,
    TimelineStepLog,
    MissionSimulationSummary,
)


def test_anomaly_campaign_definition():
    """Verifies that the standard anomaly stress campaign contains all 6 representative deep-space hazards."""
    simulator = MissionEventSimulator()
    campaign = simulator.define_standard_anomaly_campaign()

    assert len(campaign) == 6
    types = [e.event_type for e in campaign]
    assert "THRUSTER_LEAK" in types
    assert "SENSOR_BLINDNESS" in types
    assert "COLLISION_HAZARD" in types
    assert "MANIFOLD_DIVERGENCE" in types
    assert "SAFE_HARBOR_HOLD" in types
    assert "GROUND_REINSERTION" in types


def test_simulation_execution_autonomous_vs_ground():
    """Verifies that autonomous guidance survives all hazards while ground control fails at Day 40."""
    simulator = MissionEventSimulator()
    summary = simulator.run_simulation(duration_days=90.0, sample_step_days=1.0)

    # Autonomous system survived 100% of the mission
    assert summary.autonomous_mission_survived is True
    assert summary.autonomous_catastrophes_avoided >= 1
    assert summary.dro_safe_harbor_duration_days == 14.0

    # Ground-dependent architecture suffered catastrophic loss at Day 40
    assert summary.ground_dependent_mission_survived is False
    assert summary.ground_dependent_failure_epoch_day == 40.0
    assert "GROUND_DELAY_FATALITY" in summary.ground_dependent_failure_mode

    # During DRO hold (Days 56-69), daily fuel consumption must be strictly 0.00 kg
    dro_logs = [log for log in summary.step_logs if 57.0 <= log.epoch_day <= 69.0]
    for log in dro_logs:
        assert log.fuel_expended_kg == 0.0
        assert "DRO_PASSIVE_PARKING" in log.autonomous_action_taken

    # Reinsertion at Day 70 restored science operations
    reinsertion_log = [log for log in summary.step_logs if log.epoch_day == 70.0][0]
    assert "REINSERTION_EXECUTION" in reinsertion_log.autonomous_action_taken
    assert reinsertion_log.fuel_expended_kg > 10.0  # Expended delta-V fuel to reinsert
