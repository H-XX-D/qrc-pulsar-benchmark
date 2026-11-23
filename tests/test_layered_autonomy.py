import json
from dataclasses import replace
import numpy as np
import pytest
from qrc_pulsar_bench.adaptive_control import MoonCartesianFlightModel
from qrc_pulsar_bench.loss_of_track import LossOfTrackFilter
from qrc_pulsar_bench.navigation_recovery import ManeuverAwareOrbitFilter
from qrc_pulsar_bench.propulsion_guard import PropulsionGuard,PropulsionSample,GuardedPlant
from qrc_pulsar_bench.ground_authority import GroundAuthority
from qrc_pulsar_bench.protective_planner import ProtectivePlanner,ProtectiveSettings
from qrc_pulsar_bench.autonomy_supervisor import AutonomousMPC,HealthMonitor
from qrc_pulsar_bench.jacobi_control import Actuator
from qrc_pulsar_bench.compound_benchmark import CompoundExperiment
from qrc_pulsar_bench.compound_faults import Fault,FaultCase
from qrc_pulsar_bench.resilience_benchmark import Scenario


KEY=b'unit-test-key-only-never-for-flight'
STATE=np.array([1e7,0.,0.,0.,0.,0.])


def model():
    return MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),
        lambda t:(np.array([1e8,0.,0.]),np.zeros(3),np.zeros(3)),
        moon_radius_m=1.,earth_radius_m=1.,max_step_s=1800.)


def estimate(t=0.,state=STATE):
    return dict(state=np.r_[state,np.zeros(3)],covariance=np.diag([1e-6]*3+[1e-12]*6),
        time_s=t,last_observation_s=t,navigation_trusted=True,mass=500.)


def message(authority,sequence=1,action='hold_mission',**changes):
    body=dict(mission=authority.mission,sequence=sequence,issued_s=10.,not_before_s=20.,expires_s=100.,action=action,value=None)
    body.update(changes);return authority.sign(body)


def test_persistent_rejection_can_recover_without_received_packet_gap():
    m=model();r=np.diag([.1**2]*3+[.0001**2]*3)
    old=ManeuverAwareOrbitFilter(m,STATE,r,500.)
    new=LossOfTrackFilter(m,STATE,r,500.)
    for f in (old,new):f.observe(STATE,r,0.,0.)
    true=STATE.copy();true[4]=.015
    for t in np.arange(1800.,23400.,1800.):
        z=true.copy();z[:3]+=true[3:]*t
        old.observe(z,r,t,t);new.observe(z,r,t,t)
    assert not old.navigation_trusted and old.reacquisitions==0
    assert new.navigation_trusted and new.reacquisitions==1
    assert new.events[-1]['reason']=='persistent_rejection'
    assert new.events[-1]['confirmations']>=5
    assert new.events[-1]['measurement_s']-new.events[-1]['first_s']>=7200
    np.testing.assert_allclose(new.state[:6],z,atol=1e-7)


def test_replayed_packet_cannot_arm_persistent_recovery():
    r=np.eye(6);f=LossOfTrackFilter(model(),STATE,r,500.);f.observe(STATE,r,0.,0.)
    with pytest.raises(ValueError):f.observe(STATE,r,0.,20000.)
    assert f.rejection_start_s is None and not f.persistent_recovery


def test_propulsion_requires_two_indications_and_physical_confirmation():
    guard=PropulsionGuard()
    for t in (60.,120.,180.):guard.observe(PropulsionSample(t,.2,0.,False),available_s=t,expected_thrust_n=0.)
    assert not guard.isolation_requested
    for t in (240.,300.):guard.observe(PropulsionSample(t,.2,1e-4,False),available_s=t,expected_thrust_n=0.)
    assert guard.isolation_requested and not guard.thrust_allowed
    for t in (360.,420.):guard.observe(PropulsionSample(t,0.,0.,False),available_s=t,expected_thrust_n=0.)
    assert not guard.use_backup
    for t in (480.,540.):guard.observe(PropulsionSample(t,0.,0.,True),available_s=t,expected_thrust_n=0.)
    assert guard.use_backup and guard.thrust_allowed
    assert not guard.observe(PropulsionSample(540.,.2,1e-4,False),available_s=600,expected_thrust_n=0)
    assert guard.use_backup


@pytest.mark.parametrize('available',[True,False])
def test_isolation_changes_force_only_if_independent_hardware_works(available):
    guard=PropulsionGuard();plant=GuardedPlant(guard,42,isolation_available=available)
    flow=.2/(220*9.80665)
    result,cutoff=plant.propagate(model(),STATE,0.,600.,500.,None,None,np.array([.2,0,0]),flow)
    # Two valid indications isolate at120s. No control callback sees the force schedule.
    duration=120. if available else 600.
    assert result.at(600.)[6]==pytest.approx(500-flow*duration,abs=1e-9)
    assert plant.closed==available and guard.use_backup==available
    assert (cutoff==120.) if available else cutoff is None


def test_ground_auth_replay_expiry_limits_and_restart(tmp_path):
    path=tmp_path/'authority.json';a=GroundAuthority('mission',KEY,path=path)
    packet=message(a)
    assert a.receive(packet,15.)==(False,'epoch')
    wrong=dict(packet,mac='0'*64);assert a.receive(wrong,30.)==(False,'authentication')
    assert a.receive(packet,30.)==(True,'accepted') and not a.state['mission_enabled']
    restored=GroundAuthority('mission',KEY,path=path)
    assert restored.receive(packet,40.)==(False,'replay')
    assert restored.receive(message(a,2,'resume_mission'),101.)==(False,'epoch')
    assert not restored.receive(message(a,2,'set_daily_limit',value=2.),40.)[0]
    assert restored.receive(message(a,2,'resume_mission'),40.)[0]
    saved=json.loads(path.read_text());saved['body']['state']['sequence']=0;path.write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='authentication'):GroundAuthority('mission',KEY,path=path)


def test_ground_storage_failure_does_not_apply_unrecorded_authority(tmp_path,monkeypatch):
    import qrc_pulsar_bench.ground_authority as module
    a=GroundAuthority('mission',KEY,path=tmp_path/'authority.json')
    monkeypatch.setattr(module.os,'replace',lambda *args:(_ for _ in ()).throw(OSError('disk')))
    assert not a.receive(message(a),30.)[0]
    assert a.state['sequence']==0 and a.state['mission_enabled']


def test_protective_candidate_avoids_synthetic_impact_with_uncertainty_screen():
    m=model();m.max_step_s=10.;initial=np.array([80.,0.,0.,-.1,0.,0.])
    # A stationary desired reference; no retargeting to the actual falling state.
    plan=m.propagate(np.array([100.,0,0,0,0,0]),0,2000.,500.)
    settings=ProtectiveSettings(horizon_s=1500,moon_floor_m=20,earth_floor_m=1)
    planner=ProtectivePlanner(m,plan,Actuator(),settings)
    e=estimate(state=initial);dv=planner.command(0.,e,500.,1.)
    assert planner.last_report['feasible'] and np.linalg.norm(dv)>0
    coast=m.propagate(initial,0,1500,500.,floors=(20.,1.));assert not coast.completed
    burn=planner.actuator.command(dv,0,500.,interval_s=1800.)
    controlled=m.propagate(initial,0,1500,500.,burn,floors=(20.,1.));assert controlled.completed
    assert planner.last_report['ensemble_points']==13
    e['navigation_trusted']=False
    np.testing.assert_array_equal(planner.command(0,e,500.,1.),0)
    assert not planner.last_report['feasible']


def test_ground_hold_and_resume_cannot_override_navigation_or_propulsion():
    m=model();plan=m.propagate(STATE,0,20000,500.);guard=PropulsionGuard();a=GroundAuthority('m',KEY)
    c=AutonomousMPC(m,plan,guard,a);e=estimate()
    a.state['mission_enabled']=False
    np.testing.assert_array_equal(c.command(0,e,500),0)
    a.state['mission_enabled']=True;e=estimate(1800);e['navigation_trusted']=False
    np.testing.assert_array_equal(c.command(1800,e,500),0)
    assert c.last_status==2 and c.health.states['mission']=='enabled'
    guard.isolation_requested=True;e=estimate(3600)
    np.testing.assert_array_equal(c.command(3600,e,500),0)
    assert c.last_status==1024 and c.health.states['navigation']=='trusted'
    assert c.health.updates==3


def test_protective_coast_preserves_fuel_and_rearm_requires_continuous_trust():
    m=model();plan=m.propagate(STATE,0,20000,500.);guard=PropulsionGuard();a=GroundAuthority('m',KEY)
    c=AutonomousMPC(m,plan,guard,a);c.routine.recovery_required=True
    np.testing.assert_array_equal(c.command(0,estimate(),500),0)
    assert c.mode=='protective_screened_coast' and c.routine.recovery_required
    e=estimate(1800);e['navigation_trusted']=False;c.command(1800,e,500)
    assert c.inside_since is None
    for t in (3600.,5400.,7200.,9000.):
        c.command(t,estimate(t),500);assert c.routine.recovery_required
    c.command(10800.,estimate(10800.),500)
    assert not c.routine.recovery_required and not c.issued


def test_protective_planner_does_not_spend_on_feasible_coast():
    m=model();plan=m.propagate(STATE,0,86400,500.)
    planner=ProtectivePlanner(m,plan,Actuator(),ProtectiveSettings(moon_floor_m=1,earth_floor_m=1))
    e=estimate();e['state'][1]=6000.
    np.testing.assert_array_equal(planner.command(0,e,500,1.,np.array([0,-.1,0])),0)
    assert planner.last_report['reason']=='screened_coast' and planner.last_report['rollouts']==13


def test_concurrent_faults_continue_monitoring_and_ground_reports(monkeypatch,tmp_path):
    m=model();plan=m.propagate(STATE,0,86400,500.)
    exp=CompoundExperiment('.','.','.','.')
    monkeypatch.setattr(exp,'prepare',lambda scenario:(STATE.copy(),m,plan,{}))
    scenario=Scenario('unit',1,'unused',1.,18001,'unit',pulses=())
    case=FaultCase('combined','concurrent','thrust plus outages',(
        Fault('uncommanded_thrust',43200,64800,.2),Fault('sensor_dropout',43200,54000),Fault('ground_blackout',43200,64800)))
    row=exp.run(scenario,case,layered=True,journal_path=tmp_path/'journal.jsonl',ground_schedule=(
        dict(sequence=1,issued_s=36000,delivery_s=36000,expires_s=37000,action='hold_mission'),
        dict(sequence=2,issued_s=65000,delivery_s=65000,expires_s=70000,action='resume_mission')))
    assert row['metrics']['completed'] and row['protection']['primary_closed']
    assert row['protection']['health_updates']==48
    assert row['protection']['propulsion_events'][0]['time_s']==43320
    assert any(e['subsystem']=='navigation' and e['state']=='untrusted' for e in row['protection']['health_events'])
    assert row['telemetry']['pending']==0 and row['telemetry']['delivered']==row['telemetry']['events']
    assert all(c['accepted'] for c in row['protection']['ground_commands'])
    json.dumps(row,allow_nan=False)


def test_sensor_position_bias_rejected_as_unphysical_jump():
    m=model();r=np.diag([.1**2]*3+[.0001**2]*3)
    f=LossOfTrackFilter(m,STATE,r,500.)
    f.observe(STATE,r,0.,0.)
    for t in np.arange(1800.,23400.,1800.):
        z=STATE.copy();z[0]+=1000.
        f.observe(z,r,t,t)
    assert not f.navigation_trusted
    assert f.reacquisitions==0
    assert f.candidate_resets>=1
    res_true=f.observe(STATE,r,25200.,25200.)
    assert res_true['accepted'] and f.navigation_trusted
    np.testing.assert_allclose(f.state[:6],STATE,atol=1e-5)


def test_screened_recovery_burn_executed_when_safe_and_improving():
    m=model();plan=m.propagate(STATE,0,86400,500.)
    planner=ProtectivePlanner(m,plan,Actuator(),ProtectiveSettings(moon_floor_m=1,earth_floor_m=1))
    e=estimate();e['state'][1]=6000.
    proposal=np.array([0.,-.05,0.])
    dv=planner.command(0,e,500,1.,proposal,allow_recovery=True)
    assert planner.last_report['feasible']
    assert planner.last_report['reason']=='screened_burn'
    np.testing.assert_allclose(dv,proposal)

