"""
=====================================================================================
Automated Verification Suite for the Relational 3-Body Dynamical Engine
=====================================================================================
Tests time-dependent orbital evolution, energy conservation, shape-sphere invariance,
gauge elimination (zero COM drift), and trajectory consistency with Cartesian ODEs.
=====================================================================================
"""

import pytest
import numpy as np
from qrc_pulsar_bench.systems import PSR_J0337_1715
from qrc_pulsar_bench.constants import G, M_sun, day_s
from qrc_pulsar_bench.dynamics import (
    RelationalState,
    RelationalThreeBodyIntegrator,
    CartesianThreeBodyIntegrator,
    DynamicalBenchmarkSuite,
    orbital_elements_to_vectors
)

def test_relational_state_energy_and_momentum_equivalence():
    m1 = 1.4378 * M_sun
    m2 = 0.1975 * M_sun
    m3 = 0.4104 * M_sun

    r1 = np.array([4.77e9, 0.0, 0.0])
    v1 = np.array([0.0, 2.13e5, 0.0])
    r2 = np.array([0.0, 1.76e11, 0.0])
    v2 = np.array([-3.8e4, 0.0, 0.0])

    state = RelationalState(m1, m2, m3, r1, v1, r2, v2)
    x1, v1_c, x2, v2_c, x3, v3_c = state.to_barycentric_cartesian()

    # 1. COM-frame reconstruction has only floating-point roundoff
    M_tot = m1 + m2 + m3
    com_pos = (m1 * x1 + m2 * x2 + m3 * x3) / M_tot
    com_vel = (m1 * v1_c + m2 * v2_c + m3 * v3_c) / M_tot
    assert np.linalg.norm(com_pos) < 1e-5
    assert np.linalg.norm(com_vel) < 1e-10

    # 2. Kinetic energy in Jacobi coordinates must match Cartesian kinetic energy
    T_cart = 0.5 * (m1 * np.dot(v1_c, v1_c) + m2 * np.dot(v2_c, v2_c) + m3 * np.dot(v3_c, v3_c))
    mu1 = (m1 * m2) / (m1 + m2)
    mu2 = ((m1 + m2) * m3) / M_tot
    T_jac = 0.5 * (mu1 * np.dot(v1, v1) + mu2 * np.dot(v2, v2))
    assert abs(T_cart - T_jac) / T_cart < 1e-14

    # 3. Total angular momentum must match
    J_cart = m1 * np.cross(x1, v1_c) + m2 * np.cross(x2, v2_c) + m3 * np.cross(x3, v3_c)
    J_jac = state.total_angular_momentum()
    assert np.linalg.norm(J_cart - J_jac) / np.linalg.norm(J_cart) < 1e-14

def test_shape_sphere_invariance_throughout_trajectory():
    res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=5.0, num_output_points=200, rtol=1e-10, atol=1e-10
    )
    # Norm on S² must be 1.0 to machine precision along the entire trajectory
    assert res["sphere_invariance_err"] < 1e-14

    # Unsigned w3 remains in the upper hemisphere (not an oriented trajectory).
    assert res["w3"].min() >= 0.0
    assert res["w3"].max() > 0.01

def test_energy_conservation_during_time_evolution():
    res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=10.0, num_output_points=300, rtol=1e-10, atol=1e-10
    )
    # Mechanical energy must be conserved within adaptive ODE tolerance
    assert res["energy_conservation_err"] < 1e-8

def test_center_of_mass_drift_elimination():
    bench = DynamicalBenchmarkSuite.compare(
        PSR_J0337_1715, num_orbits=5.0, num_output_points=200, tolerance=1e-10
    )
    # Both are measured; do not assert a strict ranking of floating-point noise.
    assert 0 <= bench["relational_com_drift_m"] < 1e-3
    assert 0 <= bench["cartesian_com_drift_m"] < 1e-3

def test_trajectory_consistency_relational_vs_cartesian():
    num_orbits = 5.0
    rel_res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=num_orbits, num_output_points=250, rtol=1e-10, atol=1e-10
    )
    cart_res = CartesianThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=num_orbits, num_output_points=250, rtol=1e-10, atol=1e-10
    )

    # Complete vectors, including phase/orientation, not only their norms.
    cartesian = cart_res['sol_y']
    inner = cartesian[6:9]-cartesian[:3]
    f1 = PSR_J0337_1715.m1_msun/(PSR_J0337_1715.m1_msun+PSR_J0337_1715.m2_msun)
    outer = cartesian[12:15]-f1*cartesian[:3]-(1-f1)*cartesian[6:9]
    for measured, expected in [(inner,rel_res['r1_m']),(outer,rel_res['r2_m'])]:
        error = np.max(np.linalg.norm(measured-expected,axis=0))
        assert error/np.linalg.norm(expected[:,0]) < 1e-7
    assert cart_res['energy_conservation_err'] < 1e-8
    assert cart_res['angular_momentum_conservation_err'] < 1e-8


def test_pulsar_roemer_delay_has_nonzero_range():
    res = RelationalThreeBodyIntegrator.integrate(
        PSR_J0337_1715, num_inner_orbits=3.0, num_output_points=150
    )
    roemer = res["roemer_delay_us"]
    # Roemer delay must vary smoothly over the orbit with multi-second amplitude
    assert np.ptp(roemer) > 1.0e6 # Peak-to-peak > 1 second

# Independent analytic checks and adversarial regressions for the dynamics audit.
from dataclasses import replace
from types import SimpleNamespace
import qrc_pulsar_bench.dynamics as dynamics
from qrc_pulsar_bench.constants import c
from qrc_pulsar_bench.shape_sphere import ShapeSphereReduction


def test_kepler_elements_vis_viva_and_angular_momentum():
    a,e,gm = 7e9,0.42,2e20
    position,velocity = orbital_elements_to_vectors(a,e,0.8,0.3,0.7,1.2,gm)
    radius = a*(1-e*e)/(1+e*np.cos(1.2))
    assert np.linalg.norm(position) == pytest.approx(radius)
    assert np.dot(velocity,velocity) == pytest.approx(gm*(2/radius-1/a))
    assert np.linalg.norm(np.cross(position,velocity)) == pytest.approx(np.sqrt(gm*a*(1-e*e)))


def test_jacobi_forces_match_independent_pairwise_newton_sum():
    masses=np.array([3.,5.,7.])*1e25
    st=RelationalState(*masses,[2e7,3e7,-1e7],[20.,-30.,40.],[8e7,-6e7,7e7],[-4.,6.,8.])
    parts=st.to_barycentric_cartesian()
    positions=np.array(parts[::2])
    acceleration=np.zeros((3,3))
    for i in range(3):
        for j in range(3):
            if j != i:
                displacement=positions[j]-positions[i]
                acceleration[i] += G*masses[j]*displacement/np.dot(displacement,displacement)**1.5
    expected_inner=acceleration[1]-acceleration[0]
    expected_outer=acceleration[2]-(masses[0]*acceleration[0]+masses[1]*acceleration[1])/sum(masses[:2])
    y=np.concatenate([st.r1,st.v1,st.r2,st.v2])
    derivative=RelationalThreeBodyIntegrator._jacobi_derivatives(0,y,*masses)
    np.testing.assert_allclose(derivative[3:6],expected_inner,rtol=1e-14,atol=1e-15)
    np.testing.assert_allclose(derivative[9:12],expected_outer,rtol=1e-14,atol=1e-15)


def test_newtonian_two_body_limit_against_analytic_circular_trajectory():
    # Negligible, remote third mass reduces inner motion to the analytic Kepler solution.
    triple=replace(PSR_J0337_1715,m3_msun=1e-15,Pb2_days=1e7,e1=0.,e2=0.,cos_i1=1.,cos_i2=1.)
    result=RelationalThreeBodyIntegrator.integrate(triple,2,101,rtol=1e-12,atol=1e-12)
    phase=2*np.pi*result['t_eval_s']/(triple.Pb1_days*day_s)
    radius=(G*(triple.m1_msun+triple.m2_msun)*M_sun*(triple.Pb1_days*day_s/(2*np.pi))**2)**(1/3)
    expected=radius*np.array([np.cos(phase),np.sin(phase),np.zeros(len(phase))])
    assert np.max(np.linalg.norm(result['r1_m']-expected,axis=0))/radius < 1e-9


def test_roemer_projection_uses_fixed_observer_and_counts_inclination_once():
    edge=replace(PSR_J0337_1715,cos_i1=0.,cos_i2=0.)
    result=RelationalThreeBodyIntegrator.integrate(edge,1,101)
    expected=-result['cartesian_y'][2]/c*1e6
    np.testing.assert_allclose(result['roemer_delay_us'],expected,rtol=1e-14)
    assert np.ptp(expected)>3e6  # The old double tilt produced ~4.7e-10 us.
    face=replace(edge,cos_i1=1.,cos_i2=1.)
    result=RelationalThreeBodyIntegrator.integrate(face,1,30)
    np.testing.assert_array_equal(result['roemer_delay_us'],0.)


def test_geometric_shape_has_same_body_labels_in_both_apis():
    st=RelationalState(3.,5.,7.,[2.,3.,1.],[0.,0.,0.],[7.,-1.,2.],[0.,0.,0.])
    x1,_,x2,_,x3,_=st.to_barycentric_cartesian()
    expected=ShapeSphereReduction.compute_shape_coordinates(np.linalg.norm(x2-x1),np.linalg.norm(x3-x2),np.linalg.norm(x1-x3))
    np.testing.assert_allclose(st.compute_shape_coordinates()[:3],expected,atol=1e-14)


def test_com_metric_is_computed_from_returned_positions():
    result=RelationalThreeBodyIntegrator.integrate(PSR_J0337_1715,1,80)
    weights=np.array([PSR_J0337_1715.m1_msun,PSR_J0337_1715.m2_msun,PSR_J0337_1715.m3_msun])*M_sun
    weights/=weights.sum()
    y=result['cartesian_y']
    com=sum(weights[i]*y[i*6:i*6+3] for i in range(3))
    expected=np.max(np.linalg.norm(com-com[:,[0]],axis=0))
    # Both values are rounding noise of ~1e11 m positions; summation order
    # differs between implementations and platforms, so compare at that floor.
    floor=64*np.finfo(float).eps*np.max(np.abs(y[[0,1,2,6,7,8,12,13,14]]))
    assert abs(result['com_drift_m']-expected) <= floor
    assert expected < floor


def test_benchmark_detects_equal_radius_but_rotated_trajectory(monkeypatch):
    original=CartesianThreeBodyIntegrator.integrate
    rotation=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]])
    def rotated(*args,**kwargs):
        result=original(*args,**kwargs)
        for start in range(0,18,3):
            result['sol_y'][start:start+3]=rotation @ result['sol_y'][start:start+3]
        return result
    monkeypatch.setattr(CartesianThreeBodyIntegrator,'integrate',rotated)
    result=DynamicalBenchmarkSuite.compare(PSR_J0337_1715,1,30)
    assert result['max_inner_relative_difference']>0.5
    assert result['max_outer_relative_difference']>0.5


@pytest.mark.parametrize('integrator',[RelationalThreeBodyIntegrator,CartesianThreeBodyIntegrator])
@pytest.mark.parametrize('failure',['failed','partial','nonfinite'])
def test_rejects_failed_or_incomplete_solutions(monkeypatch,integrator,failure):
    def fake_solver(*args,**kwargs):
        times=kwargs['t_eval']
        if failure=='partial':
            times=times[:2]
        y=np.column_stack([args[2]]*len(times))
        if failure=='nonfinite':
            y[0,0]=np.nan
        return SimpleNamespace(success=failure!='failed',message='Injected failure',t=times,y=y,nfev=1)
    monkeypatch.setattr(dynamics,'solve_ivp',fake_solver)
    with pytest.raises(RuntimeError,match='failed|incomplete|nonfinite'):
        integrator.integrate(PSR_J0337_1715,1,10)


@pytest.mark.parametrize('options',[
    {'num_inner_orbits':0},{'num_inner_orbits':float('nan')},{'num_output_points':1},
    {'num_output_points':2.5},{'num_output_points':True},{'rtol':0},{'atol':-1},
    {'line_of_sight':[0,0,0]},{'inner_angles_rad':[0,0]},{'outer_angles_rad':[0,np.inf,0]},
])
def test_invalid_integration_inputs(options):
    with pytest.raises(ValueError):
        RelationalThreeBodyIntegrator.integrate(PSR_J0337_1715,**options)


@pytest.mark.parametrize('changes',[{'m1_msun':-1},{'Pb1_days':0},{'e1':1},{'cos_i2':1.1}])
def test_invalid_physical_inputs(changes):
    with pytest.raises(ValueError):
        RelationalThreeBodyIntegrator.integrate(replace(PSR_J0337_1715,**changes),1,10)


def test_no_observed_sep_or_spin_metadata_enters_time_evolution():
    altered=replace(PSR_J0337_1715,name='Synthetic renamed input',literature_ref='not a data source',
                    f0_hz=7.,delta_sep_published=123.,delta_sep_uncertainty=456.,delta_sep_bound95=789.)
    reference=RelationalThreeBodyIntegrator.integrate(PSR_J0337_1715,1,30)
    result=RelationalThreeBodyIntegrator.integrate(altered,1,30)
    for key in ('sol_y','roemer_delay_us','energy_j'):
        np.testing.assert_array_equal(reference[key],result[key])
    # Actual physical inputs and explicit initial phases must affect the trajectory.
    changed=RelationalThreeBodyIntegrator.integrate(replace(altered,m1_msun=1.8),1,30,inner_angles_rad=(0.2,0.3,1.))
    assert np.max(np.abs(changed['sol_y']-reference['sol_y']))>1e8
    assert reference['initial_conditions']['inner_angles_rad']==[0.,0.,0.]


def test_tighter_tolerance_improves_full_vector_agreement():
    loose=DynamicalBenchmarkSuite.compare(PSR_J0337_1715,3,120,tolerance=1e-6)
    tight=DynamicalBenchmarkSuite.compare(PSR_J0337_1715,3,120,tolerance=1e-11)
    assert tight['max_inner_relative_difference'] < loose['max_inner_relative_difference']/20
    assert tight['max_inner_relative_difference'] < 1e-8
    assert tight['max_roemer_difference_us'] < 0.01


def test_collision_rejected_instead_of_dividing_by_zero():
    st=RelationalState(1.,1.,1.,[0.,0.,0.],[0.,0.,0.],[1.,0.,0.],[0.,0.,0.])
    with pytest.raises(ValueError,match='collision'):
        st.total_energy()
