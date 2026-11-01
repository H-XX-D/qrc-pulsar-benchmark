import numpy as np
import pytest
from scipy.optimize import lsq_linear
from qrc_pulsar_bench.fpga_mpc import FixedQP,compile_core,scaled_qp,float_solve,integer_oracle,MAX_N
from qrc_pulsar_bench.predictive_mpc import finite_burn_transition,MPCSettings,PredictiveMPC


@pytest.fixture(scope='module',params=[24,32])
def qp(request,tmp_path_factory):
    b=request.param
    return FixedQP(compile_core(tmp_path_factory.mktemp('mpc')/'core.so',b,b-8),b,b-8)


def test_fixed_qp_matches_integer_oracle_and_independent_optimizer(qp):
    rng=np.random.default_rng(20261031)
    for n in (3,12,36):
        matrix=np.vstack([rng.normal(size=(2*n,n))*.1,np.eye(n)])
        target=rng.normal(size=3*n)*2
        h,g,beta=scaled_qp(matrix,target);warm=np.zeros(n)
        raw=qp.encode(h,g,warm,beta,2e-4)
        status,sol,res=qp.solve(h,g,warm,beta,2e-4)
        oracle,out,ore=integer_oracle(*raw[:3],n,*raw[3:],qp.format)
        assert status==oracle==0
        np.testing.assert_array_equal(qp.format.encode(sol),out[:n])
        assert res==ore/qp.format.scale
        reference=lsq_linear(matrix,target,bounds=(-1,1),tol=1e-12)
        assert reference.success
        np.testing.assert_allclose(sol,reference.x,atol=.0003)


def test_stale_and_invalid_inputs_inhibit(qp):
    h=np.eye(3)*.5;g=np.ones(3)*.1;warm=np.zeros(3)
    assert qp.solve(h,g,warm,.1,valid=False)[0]==2
    status,solution,_=qp.solve(h,g*np.nan,warm,.1)
    assert status==1 and not solution.any()
    H,G,W,b,t=qp.encode(h,g,warm,.1,1e-3)
    H[0]=-H[0];H[1]=qp.format.scale*2
    out=np.ones(MAX_N,dtype=np.int32);res=np.zeros(1,dtype=np.int32)
    assert qp.function(H,G,W,3,b,t,1,out,res)==1
    assert not out.any()


def test_finite_burn_response_matches_analytic_double_integrator():
    a=np.zeros((6,6));a[:3,3:]=np.eye(3)
    for duration in (0,100,600):
        phi,b=finite_burn_transition(a,1800,duration)
        np.testing.assert_allclose(phi[:3,3:],np.eye(3)*1800)
        np.testing.assert_allclose(b[:3],np.eye(3)*(1800-duration/2))
        np.testing.assert_allclose(b[3:],np.eye(3))


def test_prediction_has_future_control_coupling_and_bounds():
    class Plan:
        stop_s=100000
        def at(self,t):return np.zeros(7)
    class Model:
        def to_moon(self,x,t):return x
        def acceleration(self,r,v,t):return np.zeros(3)
    c=PredictiveMPC(Model(),Plan(),MPCSettings(horizon_steps=4))
    matrix,target,cap=c.build_problem(0,np.array([1000.,0,0,0,0,0]),500,np.zeros(4))
    h,g,beta=scaled_qp(matrix,target)
    assert np.linalg.norm(h[:3,3:])>0  # Later decisions affect the first one.
    status,solution,res=float_solve(h,g,np.zeros(12),beta)
    assert status==0
    assert np.max(abs(solution))<=1
    assert np.linalg.norm(solution[:3]*cap)<=c.settings.maximum_dv_m_s
    assert solution[0]<0
    assert np.linalg.norm(solution[3:])>0


def test_bad_solver_budget_does_not_publish_unsafe_iterate(qp):
    # Ill conditioning and nearly zero stopping tolerance force a visible failure.
    h=np.diag([.9,.00002,.00004]);g=np.array([.1,.0002,-.0003])
    status,solution,_=qp.solve(h,g,np.zeros(3),0,tolerance=1e-9)
    assert status==16 and not solution.any()
