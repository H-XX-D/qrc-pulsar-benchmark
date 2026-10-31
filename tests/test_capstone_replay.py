"""No-future-data controls and honest offline flight-comparison semantics."""
import numpy as np
import pytest
from qrc_pulsar_bench.capstone_replay import MultiDatasetFlightReplay, CapstoneFlightReplay, CapstoneReplayReport
from qrc_pulsar_bench.cislunar_forces import GM_MOON

pytest.importorskip('astropy')


@pytest.mark.parametrize('dataset,n,days', [
    ('capstone_nov2022',361,30), ('capstone_feb2023',337,28),
    ('artemis1_nov2022',121,10), ('lro_jan2023',2161,3)])
def test_archives_and_diagnostics_are_not_force_attribution(dataset,n,days):
    replay=MultiDatasetFlightReplay(dataset)
    assert replay.n_epochs==n
    assert replay.total_duration_days==pytest.approx(days)
    diag=replay.residual_diagnostics()
    assert diag['used_by_propagator'] is False
    assert np.all(np.isfinite(diag['residual_mps2']))
    assert np.linalg.norm(diag['differencing_disagreement_mps2'])>0
    forecast=replay.run_forward(duration_days=.25)
    assert np.isfinite(forecast['rms_position_error_m'])
    assert forecast['uses_reference_feedback'] is False


@pytest.mark.parametrize('harmonics',[False,True])
def test_future_spacecraft_values_cannot_change_force_or_prediction(tmp_path,harmonics):
    original=MultiDatasetFlightReplay()
    path=tmp_path/'modified.npz'
    pos=original.pos_km.copy(); vel=original.vel_km_s.copy()
    vel[1,0]+=1e-3  # Previously changed the first-hour position by ~1.5 km.
    pos[2:]+=np.array([1e6,-2e6,3e6])
    np.savez(path,times_jd=original.times_jd,positions_km=pos,velocities_km_s=vel)
    changed=MultiDatasetFlightReplay(dataset_path=str(path),environment=original.environment)
    if harmonics:
        from qrc_pulsar_bench.lunar_gravity import LunarHarmonics,InertialLunarGravity
        c=np.zeros((3,3)); s=c.copy(); c[0,0]=1; c[2,2]=2e-5
        field=LunarHarmonics(c,s,GM_MOON,1738e3,frame='TEST',source='synthetic control')
        class Orientation:
            frame='TEST'
            def __call__(self,t):
                a=t/1e5
                return np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        original.lunar_gravity=changed.lunar_gravity=InertialLunarGravity(field,Orientation())
    for t in [0, 3600, 7200]:
        np.testing.assert_array_equal(original.eval_total_acceleration(original.initial_state[:3],t),
                                      changed.eval_total_acceleration(changed.initial_state[:3],t))
    before=original.propagate(original.initial_state,[0,3600,7200])
    after=changed.propagate(changed.initial_state,[0,3600,7200])
    np.testing.assert_array_equal(before,after)
    def poison(*args): raise AssertionError('Reference/residual accessed by propagator')
    changed.residual_diagnostics=changed.get_reference_state=poison
    np.testing.assert_array_equal(after,changed.propagate(changed.initial_state,[0,3600,7200]))


def test_sampled_two_body_orbit_has_a_false_force_gap():
    radius=1829854.519277
    omega=np.sqrt(GM_MOON/radius**3)
    errors=[]
    for dt in [120.,60.,30.]:
        times=np.arange(-5,6)*dt
        velocity=radius*omega*np.c_[-np.sin(omega*times),np.cos(omega*times),np.zeros(len(times))]
        derivative=MultiDatasetFlightReplay._differentiate(times,velocity,5)[5]
        errors.append(np.linalg.norm(derivative-[-GM_MOON/radius**2,0,0]))
        if dt==120:
            coarse=MultiDatasetFlightReplay._differentiate(times,velocity,3)[5]
            assert np.linalg.norm(coarse-[-GM_MOON/radius**2,0,0])>2e-3
            assert errors[-1]<1e-5
    assert 15 < errors[0]/errors[1] < 17


def test_nonuniform_differencing_and_invalid_epochs(tmp_path):
    t=np.array([0.,1.,3.,6.,10.])
    v=np.c_[t**2,3*t+1,t*0]
    expected=np.c_[2*t,np.full(5,3),t*0]
    np.testing.assert_allclose(MultiDatasetFlightReplay._differentiate(t,v,5),expected,atol=1e-12)
    path=tmp_path/'bad.npz'
    np.savez(path,times_jd=[2450000.,2450000.,2450001.],positions_km=np.ones((3,3)),velocities_km_s=np.ones((3,3)))
    with pytest.raises(ValueError): MultiDatasetFlightReplay(dataset_path=str(path))


def test_legacy_feedback_is_explicitly_offline_and_duration_is_not_truncated():
    legacy=CapstoneFlightReplay()
    report=legacy.run_replay(duration_days=.1,check_interval_h=6)
    assert isinstance(report,CapstoneReplayReport)
    assert report.control_mode=='offline_reference_feedback'
    assert report.uses_reference_feedback
    assert report.causal_integrity_verified is False
    assert report.ground_connection_required is None
    assert report.max_tracking_error_km>0
    with pytest.raises(ValueError): legacy.run_replay(duration_days=31)
    with pytest.raises(ValueError): legacy.advance_step(legacy.pos_m[0],legacy.vel_m_s[0],0,10,dt_sub=0)
