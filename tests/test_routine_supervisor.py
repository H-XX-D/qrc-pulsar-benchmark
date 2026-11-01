from dataclasses import replace
import numpy as np
import pytest
from qrc_pulsar_bench.routine_supervisor import RoutineMPC,RoutineLimits
from test_mpc_hardening import Model,Plan,estimate

def test_large_estimated_error_latches_without_spending_and_requires_rearm():
    c=RoutineMPC(Model(),Plan());e=estimate();e['state'][0]=6000.
    assert not c.command(0,e,500).any() and c.last_status==64
    assert not c.command(1800,estimate(1800),500).any() and c.last_status==64
    with pytest.raises(ValueError):c.rearm(2000,estimate(2000,age=4000),500)
    c.rearm(3600,estimate(3600),500)
    assert c.command(3600,estimate(3600),500).any()

def test_velocity_excursion_is_causal_guard_without_truth_inputs():
    c=RoutineMPC(Model(),Plan());e=estimate();e['state'][3]=1.
    assert not c.command(0,e,500).any() and c.recovery_required

def test_caller_cannot_bypass_age_or_cadence_checks():
    c=RoutineMPC(Model(),Plan())
    assert not c.command(5000,estimate(5000,age=5000),500,valid=True).any() and c.last_status==2
    assert c.command(6000,estimate(6000),500).any()
    assert not c.command(6001,estimate(6001),500).any() and c.last_status==1

def test_rolling_budget_rejects_whole_burn_and_expires_without_refund():
    c=RoutineMPC(Model(),Plan(),limits=replace(RoutineLimits(),issued_dv_per_day_m_s=.005))
    c.issued.append((0,.005))
    assert not c.command(1800,estimate(1800),500).any() and c.last_status==128
    assert len(c.issued)==1
    c.rearm(3600,estimate(3600),500)
    assert len(c.issued)==1
    assert not c.command(3600,estimate(3600),500).any() and c.last_status==128
    # Once the old issuance expires the command may use the freed headroom.
    c.command(86400,estimate(86400),500)
    assert all(t>0 for t,_ in c.issued)
    assert sum(d for _,d in c.issued)<=.005+1e-12
