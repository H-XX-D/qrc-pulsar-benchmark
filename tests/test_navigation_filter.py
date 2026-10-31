import numpy as np
import pytest
from qrc_pulsar_bench.navigation_filter import OrbitClockEKF


def test_velocity_dependent_force_propagates_covariance_jacobian():
    state=np.zeros(8); state[3]=2
    p=np.eye(8)
    estimator=OrbitClockEKF(state,p,lambda r,t:np.zeros(3),
        state_acceleration=lambda r,v,t:-.01*v)
    estimator.predict_to(100,max_step_s=5)
    assert estimator.state[3]==pytest.approx(2*np.exp(-1),rel=1e-9)
    assert estimator.state[0]==pytest.approx(200*(1-np.exp(-1)),rel=1e-9)
    assert estimator.covariance[3,3]==pytest.approx(np.exp(-2),rel=1e-8)


def directions():
    return np.array([[1,1,1],[1,-1,-1],[-1,1,-1],[-1,-1,1]])/np.sqrt(3)


def test_prediction_covariance_matches_exact_linear_solution():
    x=np.arange(8,dtype=float); p=np.diag(np.arange(1,9,dtype=float))
    q=.02; qc=.03; dt=10
    ekf=OrbitClockEKF(x,p,lambda r,t:np.zeros(3),acceleration_noise_psd=q,clock_drift_noise_psd=qc)
    ekf.predict_to(dt)
    F=np.eye(8); F[:3,3:6]=np.eye(3)*dt; F[6,7]=dt
    Q=np.zeros((8,8))
    for i,j,noise in [(0,3,q),(1,4,q),(2,5,q),(6,7,qc)]:
        Q[i,i]=noise*dt**3/3; Q[i,j]=Q[j,i]=noise*dt**2/2; Q[j,j]=noise*dt
    np.testing.assert_allclose(ekf.state,F@x,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(ekf.covariance,F@p@F.T+Q,rtol=1e-12,atol=1e-10)


def test_position_clock_observability_and_outlier_gate():
    truth=np.array([1000.,2000,3000,0,0,0,6000,0])
    ekf=OrbitClockEKF(np.zeros(8),np.eye(8)*1e8,lambda r,t:np.zeros(3))
    d=directions(); z=d@truth[:3]+truth[6]
    report=ekf.observe(0,z,d,np.eye(4)*.01)
    np.testing.assert_allclose(report['state'][[0,1,2,6]],truth[[0,1,2,6]],atol=1e-6)
    before=ekf.state.copy()
    report=ekf.observe(0,z+1e7,d,np.eye(4),nis_limit=25)
    assert report['accepted'] is False
    np.testing.assert_array_equal(ekf.state,before)
    assert np.min(np.linalg.eigvalsh(ekf.covariance))>=0


def test_causal_sequence_ignores_future_measurement_values():
    d=directions(); p=np.eye(8)*100
    def run(last):
        ekf=OrbitClockEKF(np.zeros(8),p,lambda r,t:np.zeros(3))
        first=ekf.observe(1,np.ones(4),d,np.eye(4))
        ekf.observe(2,np.full(4,last),d,np.eye(4))
        return first
    a,b=run(2),run(2e9)
    np.testing.assert_array_equal(a['state'],b['state'])
    np.testing.assert_array_equal(a['covariance'],b['covariance'])
    ekf=OrbitClockEKF(np.zeros(8),p,lambda r,t:np.zeros(3),time_s=10)
    with pytest.raises(ValueError): ekf.observe(9,np.zeros(4),d,np.eye(4))
