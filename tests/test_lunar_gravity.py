"""Harmonic normalization, gradient, pole, frame and native ABI checks."""
import shutil
import subprocess
from pathlib import Path
import numpy as np
import pytest
from scipy.special import lpmv,gammaln
from qrc_pulsar_bench.lunar_gravity import LunarHarmonics,InertialLunarGravity
from qrc_pulsar_bench.cislunar_forces import GM_MOON,R_MOON,j2_acceleration,point_mass_acceleration


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    compiler=shutil.which('clang++') or shutil.which('g++')
    if not compiler: pytest.skip('Native C++ compiler unavailable')
    lib=tmp_path_factory.mktemp('harmonics')/'harmonics.so'
    source=Path(__file__).resolve().parents[1]/'hardware/lunar_harmonics.cpp'
    subprocess.run([compiler,'-std=c++17','-O2','-shared','-fPIC','-Wall','-Wextra',str(source),'-o',str(lib)],check=True,capture_output=True,text=True)
    return lib


def field(degree=2):
    c=np.zeros((degree+1,degree+1)); s=c.copy(); c[0,0]=1
    return c,s


@pytest.mark.parametrize('position',[[2e6,0,0],[0,0,2e6],[0,0,-2e6],[1e-9,-1e-9,2e6],[2e6,3e6,-4e6]])
def test_j2_normalization_and_poles(position,library):
    c,s=field(); c[2,0]=-2e-4/np.sqrt(5)
    f=LunarHarmonics(c,s,GM_MOON,R_MOON,frame='TEST',source='analytic-J2')
    expected=point_mass_acceleration(position,GM_MOON)+j2_acceleration(position,GM_MOON,R_MOON,2e-4)
    np.testing.assert_allclose(f.evaluate(position)[0],expected,rtol=1e-14,atol=1e-15)
    f.use_native(library)
    np.testing.assert_allclose(f.evaluate(position)[0],expected,rtol=1e-14,atol=1e-15)


def test_gradient_against_independent_legendre_potential_and_native(library):
    c,s=field(12); rng=np.random.default_rng(42)
    for n in range(2,13):
        c[n,:n+1]=rng.normal(0,1e-5,n+1)
        s[n,1:n+1]=rng.normal(0,1e-5,n)
    f=LunarHarmonics(c,s,GM_MOON,R_MOON,frame='TEST',source='synthetic')
    def potential(r):
        rn=np.linalg.norm(r); u=r[2]/rn; lon=np.arctan2(r[1],r[0]); out=0
        for n in range(13):
            for m in range(n+1):
                normalization=np.sqrt((1 if m==0 else 2)*(2*n+1)*np.exp(gammaln(n-m+1)-gammaln(n+m+1)))
                p=(-1)**m*lpmv(m,n,u)*normalization
                out+=(R_MOON/rn)**n*p*(c[n,m]*np.cos(m*lon)+s[n,m]*np.sin(m*lon))
        return GM_MOON/rn*out
    r=np.array([1e6,-1e6,1.5e6]); h=1.
    expected=np.array([(potential(r+h*e)-potential(r-h*e))/(2*h) for e in np.eye(3)])
    a,u=f.evaluate(r)
    assert u==pytest.approx(potential(r),rel=2e-15)
    np.testing.assert_allclose(a,expected,atol=1e-9,rtol=1e-8)
    f.use_native(library)
    np.testing.assert_allclose(f.evaluate(r)[0],a,atol=1e-15,rtol=1e-14)


def test_rotation_and_mismatched_frame_rejection():
    c,s=field(); c[2,2]=2e-5
    f=LunarHarmonics(c,s,GM_MOON,R_MOON,frame='PA',source='synthetic')
    class Orientation:
        frame='PA'
        def __call__(self,t): return np.array([[0.,-1,0],[1,0,0],[0,0,1]])
    o=Orientation(); gravity=InertialLunarGravity(f,o); r=np.array([2e6,0,0])
    np.testing.assert_allclose(gravity(r,10),o(10)@f.evaluate(o(10).T@r)[0])
    o.frame='ME'
    with pytest.raises(ValueError): InertialLunarGravity(f,o)


def test_shadr_rejects_missing_normalization_and_duplicate_coefficients(tmp_path):
    p=tmp_path/'field.tab'
    header='1738,4902.8,0,2,2,1,0,0\n'
    rows='1,0,0,0,0,0\n1,1,0,0,0,0\n2,0,-0.00009,0,0,0\n2,1,0,0,0,0\n2,2,0,0,0,0\n'
    p.write_text(header+rows)
    f=LunarHarmonics.from_shadr(p,2,frame='MOON_PA_DE421')
    assert f.gm==4902.8e9 and f.radius_m==1738e3
    p.write_text(header+rows.split('2,2')[0])
    with pytest.raises(ValueError,match='missing'): LunarHarmonics.from_shadr(p,2,frame='MOON_PA_DE421')
    p.write_text(header.replace(',1,0,0',',0,0,0')+rows)
    with pytest.raises(ValueError,match='normalization'): LunarHarmonics.from_shadr(p,2,frame='MOON_PA_DE421')
    p.write_text(header+'1,0,0,0,0,0\n'+rows)
    with pytest.raises(ValueError,match='Duplicate'): LunarHarmonics.from_shadr(p,2,frame='MOON_PA_DE421')


def test_higher_degree_native_matches_python_and_invalid_calls_leave_output(library):
    c,s=field(80); c[80,80]=1e-9; s[79,24]=-1e-8
    f=LunarHarmonics(c,s,GM_MOON,R_MOON,frame='TEST',source='synthetic')
    r=np.array([1.8e6,2e5,1e5]); expected=f.evaluate(r)
    f.use_native(library)
    np.testing.assert_allclose(f.evaluate(r)[0],expected[0],rtol=1e-14,atol=1e-15)
    import ctypes
    output=np.array([1.,2,3]); potential=ctypes.c_double(123)
    assert f._native(r,-1,f.gm,f.radius_m,f._c,f._s,output,ctypes.byref(potential))==1
    np.testing.assert_array_equal(output,[1,2,3]); assert potential.value==123


def test_harmonic_field_replaces_monopole_and_j2_without_double_counting(tmp_path):
    from qrc_pulsar_bench.capstone_replay import MultiDatasetFlightReplay
    from qrc_pulsar_bench.cislunar_forces import CislunarForceParameters
    path=tmp_path/'states.npz'
    np.savez(path,times_jd=2451545+np.arange(3)/86400,
             positions_km=np.tile([2000.,0,0],(3,1)),velocities_km_s=np.zeros((3,3)))
    r=np.array([2e6,0,0]); c,s=field(); c[2,0]=-9e-5
    f=LunarHarmonics(c,s,4.9e12,1738e3,frame='TEST',source='analytic control')
    replay=MultiDatasetFlightReplay(dataset_path=str(path),
        parameters=CislunarForceParameters(gm_earth=0,gm_sun=0),
        environment=lambda t:(np.array([4e8,0,0]),np.array([1e11,0,0]),np.array([0,0,1])),
        lunar_gravity=lambda r,t:f.evaluate(r)[0])
    np.testing.assert_array_equal(replay.eval_total_acceleration(r,0),f.evaluate(r)[0])
    before=replay.residual_diagnostics()['modeled_acceleration_mps2'].copy()
    replay.lunar_gravity=lambda r,t:np.zeros(3)
    after=replay.residual_diagnostics()['modeled_acceleration_mps2']
    assert np.linalg.norm(before)>1
    np.testing.assert_array_equal(after,0)
