"""Reproducible, synthetic 90-day cislunar control campaign with sourced forces.

No actual spacecraft sensor records are used. Planetary ephemerides and lunar
gravity are physical inputs; the spacecraft, measurements and faults are explicit
simulation assumptions. The commanded reference is generated before feedback.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.interpolate import make_interp_spline

from .cislunar_forces import (CislunarForceParameters, acceleration_components)
from .cislunar_perturbations import BarycentricEnvironment, CislunarPerturbations, PerturbationOptions
from .cislunar_refinement import physics_inputs
from .lunar_gravity import LunarHarmonics, LunarOrientation, InertialLunarGravity
from .jacobi_control import (JacobiFlightModel, JacobiPredictiveController, NavigationEstimate,
                             Actuator, ControlSettings)


def make_model(physics_dir, ephemeris, gm_kernel, native_library, duration_s, degree=20, max_step_s=1800.):
    epoch = 2460310.5  # 2024-01-01 00:00 TDB; explicitly synthetic mission
    env = BarycentricEnvironment(epoch, duration_s, ephemeris, gm_kernel)
    grid = env.spline.x
    positions = env.spline(grid)
    # C4 quintic interpolation makes the origin acceleration sufficiently smooth
    # for long DOP853 integrations. C2 cubic interpolation introduced measurable
    # jerk discontinuity quadrature error in the 90-day coordinate control.
    degree_inner = 5 if len(grid) >= 6 else min(3, len(grid)-1)
    inner_spline = make_interp_spline(grid, positions[:, 0]-positions[:, 1], k=degree_inner)
    inner_spline.extrapolate = False
    def inner(t):
        if not 0 <= t <= duration_s:
            raise ValueError("Epoch outside Jacobi ephemeris")
        return inner_spline(t), inner_spline(t, 1), inner_spline(t, 2)
    orientation = LunarOrientation(epoch, duration_s,
        [physics_dir/'moon_pa_de421_1900-2050.bpc', physics_dir/'moon_080317.tf'])
    field = LunarHarmonics.from_shadr(physics_dir/'gggrx_0900c_sha.tab', degree,
        frame='MOON_PA_DE421').use_native(native_library.resolve())
    lunar = InertialLunarGravity(field, orientation)
    extra = CislunarPerturbations(env, orientation,
        PerturbationOptions(earth_j2=True, other_planets=True, first_pn=True))
    parameters = CislunarForceParameters(gm_moon=field.gm, moon_radius_m=field.radius_m,
                                         moon_j2=0, area_mass_m2_kg=.02, reflectivity=1.3)
    def acceleration(r, v, t):
        bodies, _ = env(t)
        terms = acceleration_components(r, bodies[1]-bodies[0], bodies[2]-bodies[0],
                                         (0, 0, 1), parameters)
        return lunar(r, t)+terms['earth_tide']+terms['sun_tide']+terms['srp']+extra(r, v, t)
    model = JacobiFlightModel(acceleration, inner, moon_radius_m=field.radius_m, max_step_s=max_step_s)
    return model, dict(epoch_jd_tdb=epoch, lunar_degree=degree, parameters=asdict(parameters),
        extra=asdict(extra.options), spacecraft_area_mass_classification='assumed, not measured',
        inner_interpolation=f'degree {degree_inner} spline; position, velocity, acceleration from same spline',
        physics_inputs=physics_inputs(physics_dir),
        file_sha256={str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest()
                     for p in (ephemeris, gm_kernel, native_library)})


def run_campaign(model, reference, initial, duration, *, seed, outage=False, control=True,
                 settings=ControlSettings(), actuator=Actuator()):
    """Sensors supply delayed noisy state fixes; controller never sees true state.

    Position/velocity fixes stand in for a navigation subsystem, not raw TOAs or
    a validated sensor. Delayed fixes are propagated through known finite burns.
    """
    controller = JacobiPredictiveController(model, reference, duration, actuator=actuator, settings=settings)
    rng = np.random.default_rng(seed)
    # Deliberately unmodelled non-gravitational force; no fit from future truth.
    def disturbance(r, v, t):
        radial = r/np.linalg.norm(r)
        transverse = np.cross(np.cross(r, v), r); transverse /= np.linalg.norm(transverse)
        amplitude = 2e-9 + (2e-7 if 30*86400 <= t < 33*86400 else 0.)
        return amplitude*transverse + 1e-9*radial
    truth_model = JacobiFlightModel(lambda r,v,t:model.acceleration(r,v,t)+disturbance(r,v,t),
        model.inner, moon_radius_m=model.moon_radius_m, earth_radius_m=model.earth_radius_m,
        max_step_s=model.max_step_s, rtol=model.rtol)
    actual = initial + np.array([1000., -500., 300., .02, -.01, .005])
    estimated = initial.copy()
    mass = 500.
    last_observation = 0.
    history = []
    commands = []
    times, truths, estimates, errors, records = [], [], [], [], []
    total_dv = 0.
    start_wall = time.perf_counter()
    epochs = np.r_[np.arange(0, duration, settings.decision_interval_s), duration]

    def observed_state(t):
        if t == 0:
            return initial + np.array([1000., -500., 300., .02, -.01, .005])
        for trajectory in reversed(history):
            if trajectory.start_s <= t <= trajectory.stop_s:
                return trajectory.at(t)[:6]
        raise ValueError("No past truth for synthetic observation")

    def propagate_fix(q, stamp, now, measured_mass):
        # Replay accepted commands only. Burn metadata availability was logged
        # when commanded, never inferred from future trajectory differences.
        from .maneuvers import FiniteBurn
        state = q.copy(); current = stamp; m = measured_mass
        for burn in commands:
            if burn.stop_s <= stamp or burn.start_s >= now:
                continue
            if current < burn.start_s:
                coast = model.propagate(state, current, burn.start_s, m)
                state, m = coast.at(coast.stop_s)[:6], coast.at(coast.stop_s)[6]
                current = burn.start_s
            stop = min(now, burn.stop_s)
            part = FiniteBurn(current, stop, burn.thrust_n, burn.direction_icrf, m,
                              burn.mass_flow_kg_s, current, "known command during delayed-state prediction")
            trial = model.propagate(state, current, stop, m, part)
            state, m = trial.at(stop)[:6], trial.at(stop)[6]; current = stop
        if current < now:
            trial = model.propagate(state, current, now, m)
            state = trial.at(now)[:6]
        return state

    for i, t in enumerate(epochs):
        # Fixed 10-minute sensor delivery latency. A 48-hour outage occurs in
        # the stress campaign only. Each observation is generated when available.
        sensing = not (outage and 45*86400 <= t < 47*86400)
        # Common random draws at every epoch isolate the outage intervention
        # from a different post-outage noise sequence in the paired campaign.
        noise = rng.normal(size=6)
        if sensing:
            stamp = max(0., t-600.)
            sigma = np.r_[[30.]*3, [.002]*3]
            measured = observed_state(stamp)+noise*sigma
            measured_mass = mass
            for burn in commands:
                overlap = max(0., min(t, burn.stop_s)-max(stamp, burn.start_s))
                measured_mass += overlap*burn.mass_flow_kg_s
            estimated = propagate_fix(measured, stamp, t, measured_mass)
            last_observation = stamp
        times.append(float(t)); truths.append(actual.copy()); estimates.append(estimated.copy())
        errors.append(float(np.linalg.norm(actual[:3]-reference(t)[:3])))
        if i == len(epochs)-1:
            break
        end = epochs[i+1]
        age = t-last_observation
        estimate = NavigationEstimate(estimated, t, last_observation,
                                      30.+.002*age+.5*2e-7*age**2, .002+2e-7*age)
        decision = controller.decide(estimate, mass) if control else None
        burn = None if decision is None else decision.burn
        if burn is not None:
            commands.append(burn)
            consumed = burn.mass_flow_kg_s*(burn.stop_s-burn.start_s)
            dv = actuator.exhaust_velocity*np.log(mass/(mass-consumed)); total_dv += dv
        else:
            dv = 0.
        truth = truth_model.propagate(actual, t, end, mass, burn)
        if not truth.completed:
            raise RuntimeError("Simulation terminated at body collision")
        estimated_rollout = model.propagate(estimated, t, end, mass, burn)
        actual = truth.at(end)[:6]; mass = truth.at(end)[6]
        estimated = estimated_rollout.at(end)[:6]
        history.append(truth)
        if decision is not None:
            controller.confirm_execution(decision)
            records.append(dict(time_s=float(t),mode=decision.mode,reason=decision.reason,
                dv_m_s=float(dv),burn_s=0. if burn is None else burn.stop_s-burn.start_s,
                direction=None if burn is None else list(burn.direction_icrf),
                planned_times_s=decision.planned_times_s.tolist(),planned_dv_m_s=decision.planned_dv_m_s.tolist(),
                predicted_clearance_m=decision.predicted_clearance_m,candidates=decision.candidates_evaluated,
                observation_age_s=float(age),fuel_remaining_kg=float(mass-actuator.dry_mass_kg)))
        if i % 20 == 0:
            print('campaign', seed, 'control',control,'day',round(t/86400,2),'error_m',round(errors[-1],2),flush=True)
    burn_records = [r for r in records if r['dv_m_s'] > 0]
    rapid_reversals = sum(b['time_s']-a['time_s'] < settings.minimum_burn_spacing_s-1e-6
        and np.dot(a['direction'], b['direction']) < 0 for a,b in zip(burn_records[:-1],burn_records[1:]))
    times, errors = np.array(times), np.array(errors)
    truth_array = np.array(truths)
    estimate_array = np.array(estimates)
    monthly = []
    for day in np.arange(0, duration/86400, 30):
        chosen = (times >= day*86400)&(times <= (day+30)*86400)
        monthly.append(dict(start_day=float(day),rms_tracking_m=float(np.sqrt(np.mean(errors[chosen]**2))),
            max_tracking_m=float(max(errors[chosen])),dv_m_s=sum(r['dv_m_s'] for r in records if day*86400 <= r['time_s'] < (day+30)*86400)))
    result = dict(seed=seed,control=control,sensor_outage=outage,days=duration/86400,
        rms_tracking_m=float(np.sqrt(np.mean(errors**2))),max_tracking_m=float(errors.max()),
        final_tracking_m=float(errors[-1]),total_dv_m_s=float(total_dv),fuel_used_kg=float(500-mass),
        burn_count=len(burn_records),rapid_opposite_burns=rapid_reversals,
        minimum_burn_spacing_s=None if len(burn_records)<2 else float(min(np.diff([r['time_s'] for r in burn_records]))),
        mode_counts=dict(Counter(r['mode'] for r in records)),monthly=monthly,
        navigation_rms_m=float(np.sqrt(np.mean(np.sum((estimate_array[:,:3]-truth_array[:,:3])**2,axis=1)))),
        wall_seconds=time.perf_counter()-start_wall,settings=asdict(settings),actuator=asdict(actuator),commands=records)
    arrays=dict(times_s=times,truth_jacobi_si=truth_array,estimated_jacobi_si=estimate_array,tracking_error_m=errors)
    return result, arrays


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physics-dir', type=Path, required=True)
    p.add_argument('--ephemeris', type=Path, required=True)
    p.add_argument('--gm-kernel', type=Path, required=True)
    p.add_argument('--native-library', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--days', type=float, default=90.)
    p.add_argument('--degree', type=int, default=20)
    p.add_argument('--max-step', type=float, default=1800.)
    p.add_argument('--cases', nargs='+', choices=['controlled','outage','open'], default=['controlled','outage','open'])
    args = p.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    duration = args.days*86400
    model, inputs = make_model(args.physics_dir, args.ephemeris, args.gm_kernel, args.native_library,
                               duration, args.degree, args.max_step)
    # Lunar-centred, 10,000-km radius, 45-degree inclined initial circular state.
    # A high lunar orbit avoids pretending this is an NRHO/L2 reference solution.
    radius = 1e7; speed = np.sqrt(inputs['parameters']['gm_moon']/radius)
    initial = model.from_moon(np.array([radius,0,0,0,speed/np.sqrt(2),speed/np.sqrt(2)]),0)
    print('generating independent mission reference',args.days,'days',flush=True)
    plan = model.propagate(initial, 0, duration, 500.)
    if not plan.completed:
        raise RuntimeError("Planned ballistic reference collides; supply a valid mission plan")
    reference = lambda t:plan.at(t)[:6]
    report = dict(classification='synthetic closed-loop control, not flight telemetry or FPGA validation',
        mission='90-day-capable high lunar orbit; not an NRHO/L2 stationkeeping demonstration',inputs=inputs,
        reference_min_clearance_m=plan.min_clearance_m,initial_jacobi_state=initial.tolist(),results={},
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                       for name in ('jacobi_control.py','jacobi_control_demo.py')})
    for case in args.cases:
        row, arrays = run_campaign(model, reference, initial, duration, seed=20261031,
                                   outage=case=='outage', control=case!='open')
        report['results'][case] = row
        np.savez_compressed(args.output/(case+'.npz'),**arrays)
        (args.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(case, {k:v for k,v in row.items() if k not in ('commands','settings','actuator','monthly')},flush=True)


if __name__ == '__main__':
    main()
