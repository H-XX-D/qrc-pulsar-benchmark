import numpy as np
import pytest
from qrc_pulsar_bench.maneuvers import FiniteBurn,ManeuverSchedule,read_lro_small_force_metadata
from qrc_pulsar_bench.capstone_replay import MultiDatasetFlightReplay
from qrc_pulsar_bench.cislunar_forces import CislunarForceParameters,_disc_visibility


def test_burn_integration_resolves_boundaries_and_mass_flow(tmp_path):
    path=tmp_path/'synthetic.npz'
    np.savez(path,times_jd=2451545+np.array([0,20,40])/86400,positions_km=np.tile([2000.,0,0],(3,1)),velocities_km_s=np.zeros((3,3)))
    burn=FiniteBurn(10,20,10,(0.,1.,0.),100,1,0,'independent-test-command')
    replay=MultiDatasetFlightReplay(dataset_path=str(path),environment=lambda t:(np.ones(3)*1e8,np.ones(3)*1e11,np.array([0,0,1])),
                                    maneuvers=ManeuverSchedule([burn]))
    replay.eval_total_acceleration=lambda r,t:np.zeros(3)
    times=np.array([0,5,10,15,20,25,35.])
    state=replay.propagate([2e6,0,0,0,0,0],times,max_step_s=100)
    # Exact rocket equation for constant thrust/mass flow, inertial direction.
    delta_v=10*np.log(100/90)
    assert state[4,4]==pytest.approx(delta_v,rel=1e-8)
    assert state[-1,4]==pytest.approx(delta_v,rel=1e-8)
    np.testing.assert_array_equal(state[:3,4],0)
    assert state[-1,1]-state[4,1]==pytest.approx(15*delta_v,rel=1e-8)


def test_late_command_is_rejected_and_explicit_retrospective_mode_is_distinct():
    burn=FiniteBurn(10,20,10,(1.,0.,0.),100,0,30,'postflight-calibration')
    schedule=ManeuverSchedule([burn])
    with pytest.raises(ValueError,match='not available'): schedule.validate_available(0,100)
    schedule.validate_available(0,100,allow_retrospective=True)
    with pytest.raises(ValueError,match='Overlapping'): ManeuverSchedule([burn,burn])
    with pytest.raises(ValueError,match='exhausts'): FiniteBurn(0,20,10,(1,0,0),10,1,0,'test')


def test_partial_and_annular_eclipses_are_continuous():
    assert _disc_visibility(1,2,0)==0
    assert _disc_visibility(2,1,0)==pytest.approx(.75)
    assert _disc_visibility(1,1,2)==1
    assert _disc_visibility(1,1,1)==pytest.approx(1-(2*np.pi/3-np.sqrt(3)/2)/np.pi)
    values=[_disc_visibility(.005,.01,d) for d in np.linspace(0,.02,100)]
    assert np.all(np.diff(values)>=-1e-14)


def test_fdf25_metadata_is_not_promoted_to_an_inertial_burn(tmp_path):
    pytest.importorskip('astropy')
    p=tmp_path/'sample.txt'
    p.write_text('Plan Date (UTCG): 2015 126 15:05:58  Burn Start (UTCG): 2015 124 15:19:14.2\n'
                 'Burn Stop (UTCG): 2015 124 15:19:49.8\nPlanned dV (m/s): 2.537\nAverage SC Mass (kg): 1057.105\n')
    metadata=read_lro_small_force_metadata(p,source='PDS-test-fixture')
    assert metadata['direction_available'] is False
    assert metadata['plan_epoch_jd_tdb']>metadata['event'].stop_jd_tdb
    assert (metadata['event'].stop_jd_tdb-metadata['event'].start_jd_tdb)*86400==pytest.approx(35.6,abs=1e-4)
