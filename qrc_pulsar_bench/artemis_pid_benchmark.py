"""PID tuning on synthetic vehicles initialized from definitive ARTEMIS states.

The historical spacecraft did not execute our commands. Closed-loop outcomes
therefore use a separately propagated plant and synthetic observations. Later
historical states are never used as a commanded trajectory or force residual.
"""
from dataclasses import asdict, replace
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .artemis_replay import CASES, DAY, load_arc, make_model
from .adaptive_control import CausalOrbitFilter, FilterSettings, MoonCartesianFlightModel
from .jacobi_control import Actuator
from .fpga_pid import FixedPID, FloatPID, PIDSettings


def candidates():
    return {f'tau{hours}h_i{strength:.1f}': PIDSettings(response_s=hours*3600.,integral_strength=strength)
            for hours in (1,3,6) for strength in (0.,.1,.3)}


class PIDExperiment:
    def __init__(self,data,kernels,physics,native,libraries):
        self.paths=(data,kernels,physics,native);self.libraries=libraries;self.cache={}

    def prepare(self,case_name):
        if case_name not in self.cache:
            data,kernels,physics,native=self.paths
            case=next(c for c in CASES if c['name']==case_name)
            _,states,metadata=load_arc(case,data,kernels)
            model,inputs=make_model(metadata['epoch_jd_tdb'],3*DAY,physics,kernels/'de440s.bsp',kernels/'gm_de440.tpc',native)
            initial=states[0].copy()
            # A planned model trajectory derived from t0 only. No later SPK sample.
            plan=model.propagate(initial,0,3*DAY,500.)
            if not plan.completed:raise RuntimeError('Invalid planned orbit')
            self.cache[case_name]=(initial,model,plan,metadata,inputs)
        return self.cache[case_name]

    def run(self,case_name,seed,configuration,*,arithmetic='float64',faults=False,guidance_factory=None,score_step_s=None):
        initial,model,plan,metadata,inputs=self.prepare(case_name)
        if guidance_factory is not None:
            controller=guidance_factory(model,plan);settings=controller.settings
        else:
            settings=candidates()[configuration]
            if arithmetic=='float64':controller=FloatPID(settings)
            else:
                bits=int(arithmetic);controller=FixedPID(self.libraries[bits],settings,bits=bits,fraction=bits-8)
        actuator=Actuator(minimum_dv_m_s=.0001)
        rng=np.random.default_rng(seed)
        actual=initial+np.r_[rng.normal(size=3)*300,rng.normal(size=3)*.01]
        bias=rng.normal(size=3);bias*=8e-8/np.linalg.norm(bias)
        def acceleration(r,v,t):
            tangent=np.cross(np.cross(r,v),r);tangent/=np.linalg.norm(tangent)
            pulse=3e-7*tangent if DAY<=t<1.5*DAY else 0.
            return model.acceleration(r,v,t)+bias+pulse
        plant=MoonCartesianFlightModel(acceleration,model.inner,moon_radius_m=model.moon_radius_m,
                                       max_step_s=150.,rtol=2e-13)
        nav=CausalOrbitFilter(model,initial,np.diag(np.r_[[500.]*3,[.03]*3]**2),500.)
        commands=[];history=[];samples=[];records=[];mass=500.;dv_total=0.;pid_cpu=0.;online_cpu=0.
        statuses={};start=time.process_time();h=settings.interval_s;collision=False
        if score_step_s is not None and (score_step_s<=0 or not np.isclose(h/score_step_s,round(h/score_step_s))):
            raise ValueError('Dense score step must divide the control interval')
        dense=[] if score_step_s is None else [dict(time_s=0.,error_m=float(np.linalg.norm(actual[:3]-plan.at(0)[:3])))]
        covariance=np.diag(np.r_[[30.]*3,[.002]*3]**2)
        for index,t in enumerate(np.arange(0,3*DAY+1,h)):
            noise=rng.normal(size=6);magnitude=rng.normal()*.03;pointing=rng.normal(size=3)*.005
            tic=time.process_time();stamp=max(0,t-60.)
            sensing=not(faults and 1.5*DAY<=t<1.75*DAY)
            if sensing:
                measured=actual if stamp==0 else next(p for p in reversed(history) if p.start_s<=stamp<=p.stop_s).at(stamp)[:6]
                fix=measured+noise*np.r_[[30.]*3,[.002]*3]
                if faults and t==2*DAY:fix+=np.r_[10000.,-8000.,5000.,.1,-.1,.05]
                nav.observe(fix,covariance,stamp,t,commands)
            estimate=nav.forecast(t,commands,as_of_s=t)
            x=estimate['state'][:6];target=plan.at(t)[:6]
            samples.append(dict(time_s=float(t),error_m=float(np.linalg.norm(actual[:3]-target[:3])),
                                navigation_error_m=float(np.linalg.norm(actual[:3]-x[:3]))))
            if t>=3*DAY:break
            # Restricted Jacobi and Moon-relative errors are the same vectors:
            # both states receive the same prescribed Earth-Moon translation.
            error=x[:3]-target[:3];velocity=x[3:]-target[3:]
            ff=h*(model.acceleration(target[:3],target[3:],t)-model.acceleration(x[:3],x[3:],t))
            valid=t-estimate['last_observation_s']<=2*h
            timer=time.process_time()
            dv=(controller.command(t,estimate,mass,valid=valid) if guidance_factory is not None
                else controller.step(error,velocity,ff,valid=valid))
            pid_cpu+=time.process_time()-timer
            status=controller.last_status;statuses[str(status)]=statuses.get(str(status),0)+1
            burn=actuator.command(dv,t,mass,interval_s=h) if valid and status in (0,8) else None
            # The PID core is not an emergency trajectory planner. Only the
            # simulated plant's body events check collisions in this benchmark.
            online_cpu+=time.process_time()-tic
            executed=None
            if burn is not None:
                commands.append(burn);spent=burn.mass_flow_kg_s*(burn.stop_s-burn.start_s)
                dv_total+=actuator.exhaust_velocity*np.log(mass/(mass-spent))
                executed=replace(burn,thrust_n=burn.thrust_n*(1+magnitude),
                                 direction_icrf=tuple(Rotation.from_rotvec(pointing).apply(burn.direction_icrf)))
            result=plant.propagate(actual,t,t+h,mass,executed)
            if not result.completed:collision=True;break
            if score_step_s is not None:
                for stamp in np.arange(t+score_step_s,t+h+.1*score_step_s,score_step_s):
                    dense.append(dict(time_s=float(stamp),error_m=float(np.linalg.norm(result.at(stamp)[:3]-plan.at(stamp)[:3]))))
            actual,mass=result.at(t+h)[:6],result.at(t+h)[6];history.append(result)
            records.append(dict(time_s=float(t),dv=dv.tolist(),status=status,
                                burn_s=0. if burn is None else burn.stop_s-burn.start_s))
        errors=np.array([s['error_m'] for s in samples]);steady=errors[np.array([s['time_s'] for s in samples])>=DAY/2]
        dense_errors=np.array([r['error_m'] for r in dense if r['time_s']>=DAY/2])
        return dict(case=case_name,split=metadata['case']['split'],seed=seed,faults=faults,configuration=configuration,
            arithmetic=arithmetic,classification='synthetic controlled spacecraft on ARTEMIS initial geometry; not historical control validation',
            settings=asdict(settings),actuator=asdict(actuator),model=inputs,
            rms_m=float(np.sqrt(np.mean(errors**2))),steady_rms_m=None if not len(steady) else float(np.sqrt(np.mean(steady**2))),
            max_m=float(max(errors)),final_m=float(errors[-1]),dv_m_s=float(dv_total),fuel_kg=float(500-mass),
            burns=len(commands),collision=collision,pid_cpu_s=pid_cpu,guidance_cpu_s=pid_cpu,online_cpu_s=online_cpu,
            guidance_diagnostics=getattr(controller,'diagnostics',[]),
            dense_samples=dense,dense_score_step_s=score_step_s,
            dense_steady_rms_m=None if not len(dense_errors) else float(np.sqrt(np.mean(dense_errors**2))),
            dense_steady_max_m=None if not len(dense_errors) else float(max(dense_errors)),
            total_cpu_s=time.process_time()-start,core_status_counts=statuses,fixes_rejected=nav.rejected,
            fixes_accepted=nav.accepted,bias_clips=nav.bias_clips,samples=samples,commands=records)


def select(rows):
    expected={(case,seed) for case in ('P1_sep','P2_sep') for seed in (4101,4102)}
    candidates_rows=[]
    for name in candidates():
        group=[r for r in rows if r['configuration']==name]
        if len(group)!=4 or {(r['case'],r['seed']) for r in group}!=expected or any(r['split']!='training' for r in group):
            raise ValueError('Complete training cases required; evaluation cannot select gains')
        feasible=all(not r['collision'] and r['steady_rms_m'] is not None and r['steady_rms_m']<=1000 and r['max_m']<=10000 for r in group)
        candidates_rows.append(dict(configuration=name,feasible=feasible,
            mean_dv_m_s=float(np.mean([r['dv_m_s'] for r in group])),
            worst_steady_rms_m=max(r['steady_rms_m'] if r['steady_rms_m'] is not None else 1e99 for r in group)))
    feasible=[r for r in candidates_rows if r['feasible']]
    chosen=min(feasible,key=lambda r:r['mean_dv_m_s']) if feasible else min(candidates_rows,key=lambda r:r['worst_steady_rms_m'])
    return dict(selected=chosen['configuration'],met_constraints=bool(feasible),candidates=candidates_rows)
