"""Independent coordinate control and finite-thrust emergency stress scenarios."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.integrate import solve_ivp
from .jacobi_control_demo import make_model
from .jacobi_control import Actuator, ControlSettings, NavigationEstimate, JacobiPredictiveController
from .lunar_gravity import LunarHarmonics


def validate(model, inputs, duration, physics_dir, native_library):
    radius = 1e7; speed = np.sqrt(inputs['parameters']['gm_moon']/radius)
    x = np.array([radius,0,0,0,speed/np.sqrt(2),speed/np.sqrt(2)])
    q = model.from_moon(x,0.)
    jacobi = model.propagate(q,0.,duration,500.)
    times = np.linspace(0.,duration,1001)
    def rhs(t,y):
        return np.r_[y[3:],model.acceleration(y[:3],y[3:],t)]
    # Independent coordinate integration, tighter tolerance and smaller step.
    cartesian = solve_ivp(rhs,(0.,duration),x,method='DOP853',rtol=2e-13,
        atol=np.r_[[1e-6]*3,[1e-10]*3],max_step=900.,t_eval=times)
    if not cartesian.success: raise RuntimeError(cartesian.message)
    reconstructed = np.array([model.to_moon(jacobi.at(t)[:6],t) for t in times])
    errors = np.linalg.norm(reconstructed[:,:3]-cartesian.y[:3].T,axis=1)
    fields = [LunarHarmonics.from_shadr(physics_dir/'gggrx_0900c_sha.tab',degree,
        frame='MOON_PA_DE421').use_native(native_library.resolve()) for degree in (20,120)]
    # Same vector in an arbitrary fixed body frame suffices to test multipole
    # truncation at the campaign's radius, not its orientation accuracy.
    truncation = [np.linalg.norm(fields[0].evaluate(r)[0]-fields[1].evaluate(r)[0])
                  for r in reconstructed[::10,:3]]
    report = dict(days=duration/86400.,independent_coordinate_max_position_m=float(max(errors)),
                  independent_coordinate_final_position_m=float(errors[-1]),
                  degree20_vs120_max_acceleration_m_s2=float(max(truncation)),emergency_cases=[])
    print('coordinate check',report,flush=True)

    # A physically falling low-lunar orbit. A stronger hypothetical emergency
    # actuator is compared with the small routine actuator and total failure.
    near = np.array([model.moon_radius_m+200e3,0,0,0,1500.,0])
    state = model.from_moon(near,0.)
    settings = ControlSettings(decision_interval_s=3*3600.,planning_horizon_s=3*3600.,
                                moon_floor_m=50e3,earth_floor_m=200e3)
    saved_step = model.max_step_s
    model.max_step_s = 30.
    try:
        coast = model.propagate(state,0.,3*3600.,500.,floors=(50e3,200e3))
        report['coast_floor_crossing_s'] = coast.stop_s if not coast.completed else None
        for thrust in (100.,.2,0.):
            actuator = Actuator(thrust_n=thrust,emergency_burn_s=600.,dry_mass_kg=480.)
            c = JacobiPredictiveController(model,lambda t:state,3*3600.,settings=settings,actuator=actuator)
            decision = c.decide(NavigationEstimate(state,0.,0.,10.,.001),500.)
            row = dict(thrust_n=thrust,mode=decision.mode,reason=decision.reason,
                       candidates=decision.candidates_evaluated,
                       predicted_clearance_m=float(decision.predicted_clearance_m))
            if decision.burn is not None:
                burn=decision.burn
                executed=model.propagate(state,0.,3*3600.,500.,burn,floors=(50e3,200e3))
                row.update(burn_s=burn.stop_s-burn.start_s,direction=list(burn.direction_icrf),
                    verified_horizon_complete=bool(executed.completed),
                    verified_floor_clearance_m=executed.min_clearance_m,
                    delta_v_m_s=float(actuator.exhaust_velocity*np.log(500/executed.at(executed.stop_s)[6])))
            report['emergency_cases'].append(row)
            print('emergency check',row,flush=True)
    finally:
        model.max_step_s=saved_step
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physics-dir',type=Path,required=True)
    p.add_argument('--ephemeris',type=Path,required=True)
    p.add_argument('--gm-kernel',type=Path,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--days',type=float,default=90.)
    args=p.parse_args();duration=args.days*86400
    model,inputs=make_model(args.physics_dir,args.ephemeris,args.gm_kernel,args.native_library,duration)
    result=validate(model,inputs,duration,args.physics_dir,args.native_library)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
