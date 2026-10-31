import ctypes
import numpy as np
import pytest
from qrc_pulsar_bench.fpga_pid import FixedFormat, FixedPID, FloatPID, PIDSettings, compile_core


@pytest.fixture(scope='module',params=[(16,8),(24,16),(32,24)])
def core(request,tmp_path_factory):
    bits,fraction=request.param
    path=compile_core(tmp_path_factory.mktemp('pid')/'core.so',bits,fraction)
    return FixedPID(path,bits=bits,fraction=fraction)


def test_integer_oracle_matches_native_random_sequences(core):
    fmt=core.format;rng=np.random.default_rng(9182);state=np.zeros(6,dtype=np.int32)
    config=fmt.encode(PIDSettings().normalized());out=np.zeros(3,dtype=np.int32)
    for index in range(1500):
        e=fmt.encode(rng.uniform(-10,10,3));v=fmt.encode(rng.uniform(-3,3,3));ff=fmt.encode(rng.uniform(-.2,.2,3))
        valid=index%23!=0
        expected,next_state,command=fmt.oracle(e,v,ff,config,state,valid)
        actual=core.function(e,v,ff,config,int(valid),state,out)
        assert actual==expected
        np.testing.assert_array_equal(state,next_state);np.testing.assert_array_equal(out,command)


def test_saturation_freezes_integral_and_stale_freezes_all(core):
    fmt=core.format;state=np.zeros(6,dtype=np.int32);out=np.zeros(3,dtype=np.int32)
    c=fmt.encode([1,0,.25,1,.125,.125]);e=fmt.encode([1,0,0]);z=fmt.encode([0,0,0])
    for _ in range(100):assert core.function(e,z,z,c,1,state,out)==8
    assert state[0]==0 and out[0]==-c[4]
    before=state.copy();assert core.function(e,z,z,c,0,state,out)==2
    np.testing.assert_array_equal(state,before);assert not np.any(out)


def test_arithmetic_fault_is_atomic(core):
    fmt=core.format;state=np.zeros(6,dtype=np.int32);out=np.ones(3,dtype=np.int32)
    c=fmt.encode([100,0,0,1,.1,.1]);e=fmt.encode([100,0,0]);z=fmt.encode([0,0,0])
    assert core.function(e,z,z,c,1,state,out)==4
    assert not np.any(state) and not np.any(out)


def test_negative_rounding_and_input_validation():
    fmt=FixedFormat(16,8)
    np.testing.assert_array_equal(fmt.encode([.5/256,-.5/256,1.5/256,-1.5/256]),[1,-1,2,-2])
    with pytest.raises(ValueError):fmt.encode([128.])
    with pytest.raises(ValueError):fmt.encode([np.nan])


def test_pid_constant_error_integrates_and_recovers():
    settings=PIDSettings(interval_s=60,response_s=600,integral_strength=.2,
                         derivative_filter_s=0,max_dv_m_s=.15,integral_limit_m_s=.03)
    pid=FloatPID(settings)
    first=pid.step([10,0,0],[0,0,0],[0,0,0])[0]
    for _ in range(20):last=pid.step([10,0,0],[0,0,0],[0,0,0])[0]
    assert last<first<0
    before=pid.state.copy();assert not np.any(pid.step([10,0,0],[0,0,0],[0,0,0],valid=False))
    np.testing.assert_array_equal(pid.state,before)


def test_double_integrator_discrete_closed_loop_poles():
    # Analytic discrete plant, ideal velocity measurement and no clipping.
    # This tests the stated local envelope, not nonlinear orbital stability.
    s=PIDSettings(interval_s=60,response_s=600,integral_strength=.1,derivative_filter_s=0)
    h,tau=s.interval_s,s.response_s;kp,kd,ki=1/tau**2,2/tau,s.integral_strength/tau**3
    # Integral updated before zero-order-held acceleration is applied.
    u=np.array([-kp-ki*h,-kd,-ki])
    a=np.array([[1,h,0],[0,1,0],[h,0,1]],dtype=float)
    a[:2]+=np.outer([.5*h*h,h],u)
    assert np.max(np.abs(np.linalg.eigvals(a)))<1
