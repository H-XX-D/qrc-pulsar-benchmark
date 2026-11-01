import json
import numpy as np
import pytest
from qrc_pulsar_bench.adaptive_control import MoonCartesianFlightModel
from qrc_pulsar_bench.navigation_recovery import ManeuverAwareOrbitFilter
from qrc_pulsar_bench.navigation_ingress import NavigationIngress
from qrc_pulsar_bench.event_journal import EventJournal
from qrc_pulsar_bench.compound_faults import Fault,FaultCase,suite,HOUR
from qrc_pulsar_bench.compound_benchmark import CompoundExperiment,propagate_plant
from qrc_pulsar_bench.maneuvers import FiniteBurn
from qrc_pulsar_bench.resilience_benchmark import Scenario


def model():
    return MoonCartesianFlightModel(lambda r,v,t:np.zeros(3),
        lambda t:(np.array([1e8,0,0]),np.zeros(3),np.zeros(3)),
        moon_radius_m=1.,earth_radius_m=1.,max_step_s=1800.)


INITIAL=np.array([1e7,0,0,0,0,0.]);R=np.diag([900.]*3+[4e-6]*3)


@pytest.mark.parametrize('kind',['nan','future','replay','bad_covariance'])
def test_input_rejection_preserves_state_and_inhibits_commands(kind):
    nav=ManeuverAwareOrbitFilter(model(),INITIAL,R,500.);ingress=NavigationIngress(nav)
    ingress.observe(INITIAL,R,0.,0.)
    state=nav.state.copy();root=nav._root.copy();z=INITIAL.copy();r=R.copy();stamp=1800.
    if kind=='nan':z[0]=np.nan
    elif kind=='future':stamp=1900.
    elif kind=='replay':stamp=0.
    else:r[0,0]=-1.
    result=ingress.observe(z,r,stamp,1800.)
    assert result['input_error'] and not result['accepted']
    np.testing.assert_array_equal(nav.state,state);np.testing.assert_array_equal(nav._root,root)
    assert not ingress.forecast(1800.)['navigation_trusted']
    assert ingress.observe(INITIAL,R,3600.,3600.)['accepted']
    assert ingress.forecast(3600.)['navigation_trusted']


def test_numerical_failure_is_not_disguised_as_sensor_rejection(monkeypatch):
    nav=ManeuverAwareOrbitFilter(model(),INITIAL,R,500.);ingress=NavigationIngress(nav)
    def fail(*args):raise ValueError('deliberate numerical failure')
    monkeypatch.setattr(nav,'_predict_factor',fail)
    with pytest.raises(ValueError,match='numerical failure'):ingress.observe(INITIAL,R,0.,0.)
    assert ingress.invalid_packets==0


def test_telemetry_outbox_survives_restart_and_lost_ack(tmp_path):
    path=tmp_path/'events.jsonl';journal=EventJournal(path);details={'mode':'suspect'}
    journal.publish(0,'navigation_status',details);details['mode']='tampered'
    seen={}
    def lost_ack(record):seen[record['sequence']]=record;return False
    journal.deliver(lost_ack)
    assert len(journal.pending)==1 and seen[1]['details']['mode']=='suspect'
    restored=EventJournal(path)
    def receiver(record):seen.setdefault(record['sequence'],record);return True
    restored.deliver(receiver)
    assert len(seen)==1 and not restored.pending
    assert not EventJournal(path).pending
    restored.publish(100,'new',{})
    assert restored.sequence==2
    with pytest.raises(ValueError):restored.acknowledge(999)


def test_outbox_overflow_and_corrupt_records_fail_visibly(tmp_path):
    path=tmp_path/'events.jsonl';journal=EventJournal(path,capacity=1)
    journal.publish(0,'one',{})
    with pytest.raises(OverflowError):journal.publish(1,'two',{})
    with path.open('a') as stream:stream.write('{broken')
    with pytest.raises(ValueError):EventJournal(path)


def test_overlapping_uncommanded_thrust_conserves_propellant_and_vector_acceleration():
    ve=220*9.80665
    command=FiniteBurn(0.,100.,.2,(0.,1.,0.),500.,.2/ve,0.,'unit commanded thrust')
    leak=np.array([.05,0,0]);rollout=propagate_plant(model(),INITIAL,0.,600.,500.,command,leak,.05/ve)
    m1=500.-.25/ve*100;m2=m1-.05/ve*500
    expected_v=np.array([.05,.2,0])/(.25/ve)*np.log(500/m1)+leak/(.05/ve)*np.log(m1/m2)
    final=rollout.at(600.)
    assert final[6]==pytest.approx(m2,abs=1e-9)
    np.testing.assert_allclose(final[3:6],expected_v,atol=2e-9,rtol=0)
    assert len(rollout.parts)==2 and rollout.parts[0].stop_s==100.


def test_suite_has_twenty_distinct_schedules_with_overlaps_and_sequences():
    cases=suite();assert len(cases)==20 and len({c.name for c in cases})==20
    assert [sum(c.category==kind for c in cases) for kind in ('single','concurrent','consecutive')]==[10,5,5]
    for case in cases:
        assert case.onset_s>=12*HOUR and case.end_s<=60*HOUR
        if case.category=='concurrent':assert len({e.start_s for e in case.events})==1
        if case.category=='consecutive':assert len({e.start_s for e in case.events})>=3


def experiment(monkeypatch):
    m=model();plan=m.propagate(INITIAL,0,86400.,500.)
    experiment=CompoundExperiment('.','.','.','.')
    monkeypatch.setattr(experiment,'prepare',lambda scenario:(INITIAL.copy(),m,plan,{}))
    return experiment,Scenario('unit',1,'unused',1.,17001,'unit',pulses=())


def test_ground_blackout_does_not_remove_onboard_fixes_or_change_control(monkeypatch,tmp_path):
    exp,scenario=experiment(monkeypatch)
    clean=exp.run(scenario,FaultCase('none','baseline','no faults',()),journal_path=tmp_path/'clean.jsonl')
    outage=exp.run(scenario,FaultCase('ground','single','ground loss',(Fault('ground_blackout',43200.,64800.),)),
        journal_path=tmp_path/'ground.jsonl')
    assert clean['commands']==outage['commands'] and clean['dense_samples']==outage['dense_samples']
    assert outage['telemetry']['maximum_pending']>0 and outage['telemetry']['pending']==0
    assert outage['telemetry']['delivered']==outage['telemetry']['events']
    assert outage['telemetry']['maximum_delivery_latency_s']==21600.
    json.dumps(outage,allow_nan=False)


def test_concurrent_impulse_waits_for_delayed_fix_and_nan_packet_is_contained(monkeypatch):
    exp,scenario=experiment(monkeypatch)
    clean=exp.run(scenario,FaultCase('none','baseline','no faults',()))
    fault=exp.run(scenario,FaultCase('two','concurrent','kick plus bad fix',
        (Fault('impulse',43200.,43200.,.05),Fault('sensor_nan',43200.,45000.))))
    for a,b in zip(clean['commands'],fault['commands']):
        if a['time_s']<=43200:assert a==b
    assert fault['metrics']['invalid_packets']==1
    assert fault['metrics']['completed'] and 'failure' not in fault
    bad=next(s for s in fault['samples'] if s['input_error'])
    command=next(c for c in fault['commands'] if c['time_s']==bad['time_s'])
    assert command['burn_s']==0 and command['status']==2
    json.dumps(fault,allow_nan=False)


def test_telemetry_storage_failure_cannot_stop_guidance(monkeypatch):
    exp,scenario=experiment(monkeypatch);case=FaultCase('none','baseline','no faults',())
    baseline=exp.run(scenario,case)
    def broken(*args):raise OSError('simulated storage failure')
    monkeypatch.setattr(EventJournal,'publish',broken)
    failed=exp.run(scenario,case)
    assert failed['metrics']['completed'] and failed['commands']==baseline['commands']
    assert failed['telemetry']['dropped_events']>0 and failed['telemetry']['errors']
