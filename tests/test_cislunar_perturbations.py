import numpy as np
import pytest
from qrc_pulsar_bench.cislunar_perturbations import (
    induced_tide_tensor,tensor_acceleration,eih_vectorized,CislunarPerturbations,PerturbationOptions)
from qrc_pulsar_bench.relativity import eih_correction
from qrc_pulsar_bench.cislunar_forces import GM_EARTH,GM_MOON,GM_SUN


def test_tide_gradient_and_rotation():
    body=np.array([3e8,2e8,-1e8]); r=np.array([1.4e6,1e6,8e5]); radius=1738e3;k=.024116
    Q=induced_tide_tensor(body,GM_EARTH,radius,k)
    def potential(r):
        angle=r@body/(np.linalg.norm(r)*np.linalg.norm(body))
        return k*GM_EARTH*radius**5/(np.linalg.norm(body)**3*np.linalg.norm(r)**3)*.5*(3*angle**2-1)
    expected=np.array([(potential(r+e)-potential(r-e))/2 for e in np.eye(3)])
    np.testing.assert_allclose(tensor_acceleration(r,Q),expected,rtol=1e-8,atol=1e-17)
    rot,_=np.linalg.qr(np.random.default_rng(19).normal(size=(3,3)))
    np.testing.assert_allclose(tensor_acceleration(rot@r,rot@Q@rot.T),rot@expected,rtol=1e-8,atol=1e-17)
    np.testing.assert_array_equal(tensor_acceleration(r,induced_tide_tensor(body,GM_EARTH,radius,0)),0)


def test_vectorized_eih_matches_independent_loop_including_test_particle():
    rng=np.random.default_rng(31)
    for n in (2,4,12):
        x=rng.normal(size=(n,3))*1e11; v=rng.normal(size=(n,3))*3e4
        gm=np.r_[GM_SUN,rng.uniform(1e12,1e15,n-2),0]
        expected=eih_correction(x,v,gm,gravity=1.)
        np.testing.assert_allclose(eih_vectorized(x,v,gm),expected,rtol=2e-14,atol=1e-23)
    with pytest.raises(ValueError,match='collision'): eih_vectorized(np.zeros((2,3)),np.zeros((2,3)),[GM_MOON,0])


def test_differential_perturbations_vanish_at_origin_and_tide_convention():
    class Environment:
        gm=np.array([GM_MOON,GM_EARTH,GM_SUN,1e17])
        def __call__(self,t):
            return np.array([[0.,0,0],[4e8,0,0],[0,1.5e11,0],[3e11,0,0]]),np.zeros((4,3))
        def earth_pole(self,t): return np.array([0,0,1.])
    env=Environment(); orientation=lambda t:np.eye(3)
    f=CislunarPerturbations(env,orientation,PerturbationOptions(earth_j2=True,other_planets=True))
    np.testing.assert_array_equal(f(np.zeros(3),np.zeros(3),0),0)
    r=np.array([2e6,1e5,3e5])
    f=CislunarPerturbations(env,orientation,PerturbationOptions(tide_convention='epoch_subtracted'))
    np.testing.assert_array_equal(f(r,np.zeros(3),0),0)
    full=CislunarPerturbations(env,orientation,PerturbationOptions(tide_convention='full_elastic'))
    assert np.linalg.norm(full(r,np.zeros(3),0))>0
    with pytest.raises(ValueError): PerturbationOptions(tide_convention='guess')


def test_velocity_dependent_adapter_integrates_actual_state(tmp_path):
    from qrc_pulsar_bench.capstone_replay import MultiDatasetFlightReplay
    path=tmp_path/'synthetic.npz'
    np.savez(path,times_jd=2451545+np.array([0,50,100])/86400,
             positions_km=np.tile([2000.,0,0],(3,1)),velocities_km_s=np.zeros((3,3)))
    replay=MultiDatasetFlightReplay(dataset_path=str(path),
        environment=lambda t:(np.array([4e8,0,0]),np.array([1e11,0,0]),np.array([0,0,1.])),
        additional_acceleration=lambda r,v,t:-.01*v)
    replay.eval_total_acceleration=lambda r,t:np.zeros(3)
    state=np.array([2e6,0,0,0,2.,0])
    out=replay.propagate(state,[0,50,99],max_step_s=5)[-1]
    assert out[4]==pytest.approx(2*np.exp(-.99),rel=1e-9)
    assert out[1]==pytest.approx(200*(1-np.exp(-.99)),rel=1e-9)


def test_gm_kernel_preserves_fortran_exponents_and_km_conversion(tmp_path):
    from qrc_pulsar_bench.cislunar_perturbations import read_gm_kernel
    p=tmp_path/'gm.tpc'
    p.write_text('BODY301_GM = ( 4.9028001184575496D+03 )\nBODY399_GM = ( 3.9860043550702266E+05 )\n')
    parsed=read_gm_kernel(p)
    assert parsed[301]==GM_MOON
    assert parsed[399]==GM_EARTH
