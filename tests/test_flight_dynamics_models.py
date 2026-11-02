"""Tests for flight-critical spacecraft dynamics and environmental perturbation models."""

import numpy as np
import pytest

from qrc_pulsar_bench.flight_dynamics_models import (
    OpticalSurface,
    SolarRadiationTorqueModel,
    ReactionWheel,
    ReactionWheelArray,
    PropellantTankSlosh,
    GravityGradientTorque,
    StarTrackerModel,
    EclipseThermalSnapModel,
)
from qrc_pulsar_bench.thruster_allocation import RCSCluster, ThrusterAllocator
from qrc_pulsar_bench.constants import AU, c
from qrc_pulsar_bench.cislunar_forces import GM_MOON, R_MOON


def test_solar_radiation_torque_and_cop_offset():
    # Model two solar array wings: +Y wing and -Y wing
    wing_plus_y = OpticalSurface(
        name="wing_plus_y",
        area_m2=5.0,
        normal_body=(0.0, 0.0, 1.0),
        center_of_pressure_body_m=(0.0, 2.5, 0.0),
        specular=0.2,
        diffuse=0.1,
    )
    wing_minus_y = OpticalSurface(
        name="wing_minus_y",
        area_m2=5.0,
        normal_body=(0.0, 0.0, 1.0),
        center_of_pressure_body_m=(0.0, -2.5, 0.0),
        specular=0.2,
        diffuse=0.1,
    )
    srp_model = SolarRadiationTorqueModel([wing_plus_y, wing_minus_y])

    # Spacecraft at 1 AU, Sun directly along +Z
    r_sc = np.array([0.0, 0.0, 0.0])
    r_sun = np.array([0.0, 0.0, AU])
    dcm_body_to_icrf = np.eye(3)

    # 1. Symmetric CoM at (0, 0, 0): Torque must be exactly zero by symmetry
    r_cm_sym = np.array([0.0, 0.0, 0.0])
    res_sym = srp_model.evaluate(r_sc, r_sun, dcm_body_to_icrf, r_cm_sym)

    assert res_sym.sunlight_fraction == 1.0
    # Net force should be along -Z
    assert res_sym.force_body_n[2] < 0.0
    np.testing.assert_allclose(res_sym.torque_body_nm, np.zeros(3), atol=1e-12)

    # 2. Shift CoM along +Y by 0.2 m: Wing +Y now has lever arm 2.3 m, Wing -Y has 2.7 m
    r_cm_shifted = np.array([0.0, 0.2, 0.0])
    res_shifted = srp_model.evaluate(r_sc, r_sun, dcm_body_to_icrf, r_cm_shifted)

    # Force remains the same
    np.testing.assert_allclose(res_shifted.force_body_n, res_sym.force_body_n, atol=1e-12)
    # Disturbance torque should now be non-zero around X axis
    assert abs(res_shifted.torque_body_nm[0]) > 1e-7


def test_reaction_wheel_array_orthogonal_and_pyramid():
    # 1. Orthogonal 3-wheel array
    rwa_3 = ReactionWheelArray.standard_3_axis_orthogonal(
        inertia_kg_m2=0.05, max_speed_rpm=6000.0, max_torque_nm=0.2
    )
    assert rwa_3.num_wheels == 3

    # Commanded torque [0.1, -0.05, 0.08] N*m
    cmd_tau = np.array([0.1, -0.05, 0.08])
    u_motor = rwa_3.allocate_torque(cmd_tau)
    # Bus torque tau_bus = -A_w * u_motor should match cmd_tau
    tau_bus = -rwa_3.A_w @ u_motor
    np.testing.assert_allclose(tau_bus, cmd_tau, atol=1e-12)

    # Step simulation forward for 1 second
    achieved_tau = rwa_3.step(1.0, u_motor)
    np.testing.assert_allclose(achieved_tau, cmd_tau, atol=1e-12)
    assert np.all(np.abs(rwa_3.wheel_speeds_rad_s) > 0.0)

    # 2. 4-wheel pyramid array
    rwa_4 = ReactionWheelArray.standard_4_wheel_pyramid(
        skew_angle_deg=54.7356, inertia_kg_m2=0.05, max_speed_rpm=6000.0, max_torque_nm=0.2
    )
    assert rwa_4.num_wheels == 4
    u_4 = rwa_4.allocate_torque(cmd_tau)
    tau_bus_4 = -rwa_4.A_w @ u_4
    np.testing.assert_allclose(tau_bus_4, cmd_tau, atol=1e-10)

    # Micro-vibration jitter should scale with wheel speed
    rwa_3.wheel_speeds_rad_s = np.array([300.0, 300.0, 300.0])  # ~2860 RPM
    f_jitter, tau_jitter = rwa_3.micro_vibration_jitter()
    assert np.linalg.norm(f_jitter) > 0.0
    assert np.linalg.norm(tau_jitter) > 0.0


def test_reaction_wheel_desaturation_momentum_dump():
    rwa = ReactionWheelArray.standard_3_axis_orthogonal(
        inertia_kg_m2=0.05, max_speed_rpm=6000.0, max_torque_nm=0.2
    )
    # Spin wheels up to 85% of maximum speed (saturation condition)
    max_rad_s = 6000.0 * (2.0 * np.pi / 60.0)
    rwa.wheel_speeds_rad_s = np.array([0.85 * max_rad_s, 0.2 * max_rad_s, -0.82 * max_rad_s])

    needs_dump, ratios = rwa.check_desaturation(threshold_fraction=0.8)
    assert needs_dump
    assert ratios[0] >= 0.85

    # Setup RCS cluster and allocator for momentum dumping
    cluster = RCSCluster.standard_12_thruster_cluster(radius_m=0.6, height_m=0.8, thrust_n=2.0)
    allocator = ThrusterAllocator(cluster, interval_s=1.0, torque_weight=100.0)
    r_cm = np.zeros(3)

    # Plan 30-second RCS momentum dump to zero target momentum (well within 2 N RCS authority)
    dump_plan = rwa.plan_momentum_dump(allocator, r_cm, duration_s=30.0)

    assert dump_plan.burn_duration_s == 30.0
    assert dump_plan.allocation.success
    # Achieved RCS torque should closely match required dumping torque
    np.testing.assert_allclose(
        dump_plan.allocation.achieved_wrench[3:],
        dump_plan.required_torque_nm,
        atol=0.05
    )
    # Net translational force during momentum dump should be near zero (pure couple)
    np.testing.assert_allclose(dump_plan.allocation.achieved_wrench[:3], np.zeros(3), atol=0.05)
    assert dump_plan.propellant_used_kg > 0.0


def test_propellant_tank_slosh_dynamics():
    slosh = PropellantTankSlosh(
        tank_location_body_m=(0.0, 0.0, -0.5),
        propellant_mass_kg=120.0,
        tank_radius_m=0.4,
        slosh_fraction=0.25,
        damping_ratio=0.02,
        feed_axis_body=(0.0, 0.0, 1.0),
    )
    assert slosh.slosh_mass_kg == pytest.approx(30.0)

    # Step in steady axial acceleration (e.g. 1.0 m/s^2 along Z)
    # Zero lateral acceleration -> slosh should remain near zero
    state0 = slosh.step(0.1, np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(state0.displacement_m, np.zeros(2), atol=1e-6)

    # Apply lateral acceleration disturbance (e.g. 0.2 m/s^2 along X)
    state1 = slosh.step(0.1, np.array([0.2, 0.0, 1.0]))
    assert np.linalg.norm(state1.displacement_m) > 0.0
    # Reaction force and reaction torque imparted to bus
    assert np.linalg.norm(state1.reaction_force_body_n) > 0.0
    assert np.linalg.norm(state1.reaction_torque_body_nm) > 0.0


def test_zero_g_ullage_settling_and_bubble_hazard():
    slosh = PropellantTankSlosh(
        tank_location_body_m=(0.0, 0.0, -0.5),
        propellant_mass_kg=120.0,
        feed_axis_body=(0.0, 0.0, 1.0),
    )

    # 1. Zero-g coasting: zero net acceleration
    is_settled, bubble_hazard, a_axial = slosh.check_ullage_settling(np.zeros(3))
    assert not is_settled
    assert bubble_hazard
    assert a_axial == 0.0

    # 2. Settling burn active (0.05 m/s^2 along feed axis Z)
    is_settled, bubble_hazard, a_axial = slosh.check_ullage_settling(np.array([0.0, 0.0, 0.05]))
    assert is_settled
    assert not bubble_hazard
    assert a_axial == 0.05

    # 3. Plan settling burn for 500 kg craft needing 0.1 m/s delta-v
    plan = slosh.plan_ullage_settling_burn(
        total_spacecraft_mass_kg=500.0, rcs_settling_thrust_n=4.0, required_delta_v_m_s=0.1
    )
    # a = 4 N / 500 kg = 0.008 m/s^2; t = 0.1 / 0.008 = 12.5 s
    assert plan["duration_s"] == pytest.approx(12.5, rel=1e-3)
    assert plan["propellant_used_kg"] > 0.0


def test_gravity_gradient_disturbance_torque():
    # Long cylinder aligned along Z: Ixx = Iyy = 100 kg*m^2, Izz = 20 kg*m^2
    inertia = np.diag([100.0, 100.0, 20.0])

    # Spacecraft in low lunar orbit: distance r = 1850 km
    r_dist = R_MOON + 100.0e3

    # 1. Spacecraft aligned with radius vector (r_hat = [0, 0, 1]): torque must be zero
    r_body_aligned = np.array([0.0, 0.0, r_dist])
    tau_aligned = GravityGradientTorque.compute(r_body_aligned, inertia, GM_MOON)
    np.testing.assert_allclose(tau_aligned, np.zeros(3), atol=1e-12)

    # 2. Canted 45 degrees between X and Z: should experience restoring torque
    angle = np.pi / 4.0
    r_body_canted = np.array([np.sin(angle), 0.0, np.cos(angle)]) * r_dist
    tau_canted = GravityGradientTorque.compute(r_body_canted, inertia, GM_MOON)

    # Torque must be non-zero around Y axis
    assert abs(tau_canted[1]) > 1e-6
    # Torque must be strictly orthogonal to the radius vector (r_hat . tau == 0)
    r_hat = r_body_canted / r_dist
    assert np.dot(r_hat, tau_canted) == pytest.approx(0.0, abs=1e-12)


def test_star_tracker_blinding_and_stellar_aberration():
    st = StarTrackerModel(
        boresight_body=(1.0, 0.0, 0.0),
        sun_exclusion_deg=35.0,
        moon_exclusion_deg=15.0,
        earth_exclusion_deg=20.0,
    )
    r_sc = np.zeros(3)
    dcm = np.eye(3)

    # 1. Sun directly in front of boresight (angle = 0 deg < 35 deg): blinded by sun
    r_sun = np.array([AU, 0.0, 0.0])
    status_sun = st.check_blinding(r_sc, r_sun, dcm)
    assert status_sun.is_blinded
    assert status_sun.blinded_by == "sun"
    assert status_sun.sun_angle_deg == pytest.approx(0.0, abs=1e-4)

    # 2. Sun 90 degrees away: clean pointing
    r_sun_clear = np.array([0.0, AU, 0.0])
    status_clear = st.check_blinding(r_sc, r_sun_clear, dcm)
    assert not status_clear.is_blinded
    assert status_clear.blinded_by is None
    assert status_clear.sun_angle_deg == pytest.approx(90.0, abs=1e-4)

    # 3. Thruster plume firing active: blinded by plume particles
    status_plume = st.check_blinding(r_sc, r_sun_clear, dcm, thruster_plume_active=True)
    assert status_plume.is_blinded
    assert status_plume.blinded_by == "plume"

    # 4. Relativistic stellar aberration
    star_cat = np.array([0.0, 0.0, 1.0])
    # Spacecraft velocity = 29.8 km/s along +X (Earth orbital speed)
    v_sc = np.array([29.8e3, 0.0, 0.0])
    u_app, deflection_arcsec = st.relativistic_stellar_aberration(star_cat, v_sc)

    # v / c = 29.8e3 / 299792458 = 9.94e-5 rad ~ 20.5 arcsec
    assert deflection_arcsec == pytest.approx(20.5, abs=0.5)
    assert u_app[0] > 0.0  # Shifted in direction of velocity


def test_eclipse_thermal_snap_impulse():
    snap_model = EclipseThermalSnapModel(
        boom_length_m=6.0,
        panel_mass_kg=20.0,
        thermal_expansion_coeff=23e-6,
        delta_temperature_k=200.0,
    )

    # 1. Steady sunlight (sun fraction = 1.0 -> 1.0): zero shock
    shock_steady = snap_model.compute_shock_impulse(1.0, 1.0, dt_s=1.0)
    assert shock_steady is None

    # 2. Sudden eclipse ingress: sun fraction drops from 0.8 to 0.1 in 1 second
    shock_ingress = snap_model.compute_shock_impulse(0.8, 0.1, dt_s=1.0)
    assert shock_ingress is not None
    assert isinstance(shock_ingress, np.ndarray)
    assert np.linalg.norm(shock_ingress) > 0.0
