"""Matched control tuning/evaluation with a written split and paired randomness.

All spacecraft trajectories and direct-state measurements are synthetic. Source
gravity and planetary inputs are shared. The plant is independently propagated
in Moon-centred Cartesian coordinates, including declared execution errors.
The training selector never reads evaluation outcomes.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .jacobi_control_demo import make_model
from .jacobi_control import (Actuator, ControlSettings, NavigationEstimate,
                             JacobiFlightModel, JacobiPredictiveController)
from .adaptive_control import (MoonCartesianFlightModel,CausalOrbitFilter,FilterSettings,
                               BiasAwareController,propagate_known_commands)
from .maneuvers import FiniteBurn


DAY=86400.


@dataclass(frozen=True)
class Scenario:
    name: str
    seed: int
    days: float
    radius_m: float
    inclination_deg: float
    position_sigma_m: float=30.
    velocity_sigma_m_s: float=.002
    thrust_sigma_fraction: float=.03
    pointing_sigma_rad: float=.005
    bias_m_s2: float=2e-8
    pulse_m_s2: float=2e-7
    outage_days: float=0.
    outlier: bool=False


TRAINING=(Scenario('train_a',1101,6,8e6,35),
          Scenario('train_b',1102,6,1e7,55,position_sigma_m=60.,velocity_sigma_m_s=.004,outlier=True),
          Scenario('train_c',1103,6,1.4e7,70,outage_days=.5,bias_m_s2=8e-8))
EVALUATION=(Scenario('test_a',2201,9,7e6,25),
            Scenario('test_b',2202,9,1.2e7,50),
            Scenario('test_c',2203,9,1.6e7,65,outage_days=1.),
            Scenario('test_d',2204,9,7e6,65,position_sigma_m=75.,velocity_sigma_m_s=.006,outlier=True),
            Scenario('test_e',2205,9,1.2e7,25,thrust_sigma_fraction=.08,pointing_sigma_rad=.02,bias_m_s2=1e-7),
            Scenario('test_f',2206,9,1.6e7,50,outage_days=1.,outlier=True,bias_m_s2=1e-7))
LONG=Scenario('month_holdout',3301,30,1e7,45,outage_days=2.,outlier=True)


def configurations():
    default=ControlSettings()
    # Equal eight-configuration training budget for each guidance family.
    # Expanded using training evidence only, before any evaluation was opened.
    candidates={
        'mpc_economy':('mpc',replace(default,dv_cost_scale_m_s=.025)),
        'mpc_original_weights':('mpc',default),
        'mpc_precise':('mpc',replace(default,dv_cost_scale_m_s=.1,planning_horizon_s=DAY)),
        'mpc_wide_deadband':('mpc',replace(default,position_on_m=500.,position_off_m=200.,dv_cost_scale_m_s=.05)),
        'mpc_1day_original':('mpc',replace(default,planning_horizon_s=DAY)),
        'mpc_2day_precise':('mpc',replace(default,dv_cost_scale_m_s=.1)),
        'mpc_1day_high_precision':('mpc',replace(default,dv_cost_scale_m_s=.2,planning_horizon_s=DAY)),
        'mpc_1day_balanced':('mpc',replace(default,dv_cost_scale_m_s=.075,planning_horizon_s=DAY)),
        'pd_12h':('pd',replace(default,response_time_s=12*3600.)),
        'pd_18h':('pd',replace(default,response_time_s=18*3600.)),
        'pd_24h':('pd',replace(default,response_time_s=24*3600.)),
        'pd_36h':('pd',replace(default,response_time_s=36*3600.)),
        'pd_6h':('pd',replace(default,response_time_s=6*3600.)),
        'pd_8h':('pd',replace(default,response_time_s=8*3600.)),
        'pd_10h':('pd',replace(default,response_time_s=10*3600.)),
        'pd_48h':('pd',replace(default,response_time_s=48*3600.)),
        'target_6h':('target',replace(default,planning_horizon_s=6*3600.)),
        'target_12h':('target',replace(default,planning_horizon_s=12*3600.)),
        'target_24h':('target',replace(default,planning_horizon_s=24*3600.)),
        'target_48h':('target',replace(default,planning_horizon_s=48*3600.)),
        'target_6h_economy':('target',replace(default,planning_horizon_s=6*3600.,dv_cost_scale_m_s=.025)),
        'target_6h_precise':('target',replace(default,planning_horizon_s=6*3600.,dv_cost_scale_m_s=.1)),
        'target_6h_high_precision':('target',replace(default,planning_horizon_s=6*3600.,dv_cost_scale_m_s=.2)),
        'target_12h_precise':('target',replace(default,planning_horizon_s=12*3600.,dv_cost_scale_m_s=.1)),
    }
    return candidates


class CountedForce:
    def __init__(self,force):self.force=force;self.calls=0
    def __call__(self,r,v,t):
        self.calls+=1
        return self.force(r,v,t)


class Experiment:
    def __init__(self,model,inputs):
        self.model,self.inputs=model,inputs
        self.reference_cache={}

    def reference(self,case):
        key=(case.radius_m,case.inclination_deg,case.days)
        if key not in self.reference_cache:
            speed=np.sqrt(self.inputs['parameters']['gm_moon']/case.radius_m)
            inc=np.deg2rad(case.inclination_deg)
            initial=np.array([case.radius_m,0,0,0,speed*np.cos(inc),speed*np.sin(inc)])
            model=MoonCartesianFlightModel(self.model.acceleration,self.model.inner,
                moon_radius_m=self.model.moon_radius_m,max_step_s=1800.,rtol=2e-13)
            plan=model.propagate(initial,0.,case.days*DAY,500.)
            if not plan.completed:raise RuntimeError('Invalid reference orbit')
            self.reference_cache[key]=(initial,plan)
        return self.reference_cache[key]

    def run(self,case,*,configuration='mpc_original_weights',coordinates='jacobi',navigation='filter',bias=True):
        if coordinates not in ('jacobi','cartesian') or navigation not in ('raw','filter'):
            raise ValueError('Invalid explicit comparison mode')
        method,settings=configurations()[configuration]
        initial,plan=self.reference(case);duration=case.days*DAY
        force=CountedForce(self.model.acceleration)
        cls=JacobiFlightModel if coordinates=='jacobi' else MoonCartesianFlightModel
        base=cls(force,self.model.inner,moon_radius_m=self.model.moon_radius_m,
                 earth_radius_m=self.model.earth_radius_m,max_step_s=1800.,rtol=2e-13)
        reference=lambda t:base.from_moon(plan.at(t)[:6],t)
        if navigation=='raw':
            # Frozen original logic and settings, merely exposed to identical
            # new scenario inputs and the same independent Cartesian plant.
            controller=JacobiPredictiveController(base,reference,duration,settings=settings)
        else:
            controller=BiasAwareController(base,reference,duration,settings=settings,method=method)
        actuator=controller.actuator
        rng=np.random.default_rng(case.seed)
        offset=np.r_[rng.normal(size=3)*700.,rng.normal(size=3)*.015]
        actual=initial+offset
        direction=rng.normal(size=3);direction/=np.linalg.norm(direction)
        def disturbance(r,v,t):
            tangential=np.cross(np.cross(r,v),r);tangential/=np.linalg.norm(tangential)
            episode=case.pulse_m_s2 if duration/3<=t<duration/3+min(2*DAY,duration/6) else 0.
            return case.bias_m_s2*direction+episode*tangential
        truth_model=MoonCartesianFlightModel(lambda r,v,t:self.model.acceleration(r,v,t)+disturbance(r,v,t),
            self.model.inner,moon_radius_m=self.model.moon_radius_m,max_step_s=900.,rtol=2e-13)
        raw_model=MoonCartesianFlightModel(force,self.model.inner,moon_radius_m=self.model.moon_radius_m,
                                           max_step_s=1800.,rtol=2e-13)
        sigma=np.r_[[case.position_sigma_m]*3,[case.velocity_sigma_m_s]*3];measurement_cov=np.diag(sigma*sigma)
        filter_=CausalOrbitFilter(base,initial,np.diag(np.r_[[1000.]*3,[.05]*3]**2),500.)
        raw_state=initial.copy();raw_stamp=0.;raw_mass=500.;last_observation=0.
        history=[];commands=[];mass=500.;records=[];samples=[];decisions=[];nis=[];nees=[]
        issued=0;budget_violations=0;cpu_planning=0.;cpu_navigation=0.;total_dv=0.
        max_calls=0;max_plan_cpu=0.;last_innovation=None
        started=time.process_time();wall=time.perf_counter()
        epochs=np.r_[np.arange(0.,duration,settings.decision_interval_s),duration]
        for index,t in enumerate(epochs):
            # Draw once per epoch even if no fix or burn occurs, so all variants
            # share measurement and actuator errors at matching times.
            noise=rng.normal(size=6)
            magnitude_error=rng.normal()*case.thrust_sigma_fraction
            rotation_error=rng.normal(size=3)*case.pointing_sigma_rad
            stamp=max(0.,t-600.)
            sensing=not(duration/2<=t<duration/2+case.outage_days*DAY)
            before_calls=force.calls;nav_start=time.process_time()
            if sensing:
                if stamp==0.:physical=initial+offset;measured_mass=500.
                else:
                    past=next(h for h in reversed(history) if h.start_s<=stamp<=h.stop_s)
                    y=past.at(stamp);physical,measured_mass=y[:6],y[6]
                measured=physical+noise*sigma
                if case.outlier and index==int(round(.7*duration/settings.decision_interval_s)):
                    measured+=np.array([10000.,-8000.,5000.,.1,-.1,.05])
                if navigation=='filter':
                    receipt=filter_.observe(measured,measurement_cov,stamp,t,commands)
                    nis.append(receipt['nis']);last_innovation=receipt['accepted']
                else:
                    raw_state,raw_stamp,raw_mass=measured,stamp,measured_mass
                    last_observation=stamp
            if navigation=='filter':
                forecast=filter_.forecast(t,commands,as_of_s=t)
                predicted=forecast['state'][:6];cov=forecast['covariance']
                estimated_bias=forecast['state'][6:] if bias else np.zeros(3)
                controller.set_bias(estimated_bias)
                last_observation=forecast['last_observation_s']
                sigma_r=float(np.sqrt(max(0.,np.linalg.eigvalsh(cov[:3,:3]).max())))
                sigma_v=float(np.sqrt(max(0.,np.linalg.eigvalsh(cov[3:6,3:6]).max())))
                delta=predicted-actual
                nees.append(float(delta@np.linalg.solve(cov[:6,:6],delta)))
            else:
                predicted,_,_=propagate_known_commands(raw_model,raw_state,raw_stamp,t,raw_mass,commands,as_of_s=t)
                estimated_bias=np.zeros(3)
                age=t-last_observation
                sigma_r=case.position_sigma_m+age*case.velocity_sigma_m_s+.5*2e-7*age**2
                sigma_v=case.velocity_sigma_m_s+2e-7*age
            cpu_navigation+=time.process_time()-nav_start
            samples.append(dict(time_s=float(t),error_m=float(np.linalg.norm(actual[:3]-plan.at(t)[:3])),
                navigation_error_m=float(np.linalg.norm(predicted[:3]-actual[:3])),bias=estimated_bias.tolist()))
            if index==len(epochs)-1:break
            estimate=NavigationEstimate(base.from_moon(predicted,t),t,last_observation,sigma_r,sigma_v)
            timer=time.process_time();decision=controller.decide(estimate,mass)
            elapsed=time.process_time()-timer;cpu_planning+=elapsed;max_plan_cpu=max(max_plan_cpu,elapsed)
            calls=force.calls-before_calls;max_calls=max(max_calls,calls)
            budget_violations+=int(calls>5000)
            burn=decision.burn;executed=None;command_dv=0.
            if burn is not None:
                commands.append(burn);issued+=1
                flow=burn.mass_flow_kg_s;consumed=flow*(burn.stop_s-burn.start_s)
                command_dv=actuator.exhaust_velocity*np.log(mass/(mass-consumed));total_dv+=command_dv
                pointing=Rotation.from_rotvec(rotation_error).apply(burn.direction_icrf)
                # Same mass flow, imperfect delivered impulse: no hidden mass
                # telemetry correction or truth-to-filter state reset.
                executed=replace(burn,thrust_n=burn.thrust_n*max(0.,1+magnitude_error),direction_icrf=tuple(pointing))
            end=epochs[index+1]
            truth=truth_model.propagate(actual,t,end,mass,executed)
            if not truth.completed:raise RuntimeError('Physical collision in benchmark')
            actual,mass=truth.at(end)[:6],truth.at(end)[6]
            history.append(truth);controller.confirm_execution(decision)
            records.append(dict(time_s=float(t),mode=decision.mode,dv_m_s=float(command_dv),
                magnitude_error_fraction=float(magnitude_error),pointing_error_rad=rotation_error.tolist(),
                planned_cpu_s=float(elapsed),online_force_calls=int(calls),propellant_kg=float(mass-actuator.dry_mass_kg),
                direction=None if burn is None else list(burn.direction_icrf),burn_s=0. if burn is None else burn.stop_s-burn.start_s))
            decisions.append(decision.mode)
        errors=np.array([s['error_m'] for s in samples]);steady=errors[np.array([s['time_s'] for s in samples])>=DAY]
        row=dict(scenario=asdict(case),configuration=configuration,coordinates=coordinates,navigation=navigation,bias_feedforward=bool(bias),
            rms_m=float(np.sqrt(np.mean(errors**2))),steady_rms_m=float(np.sqrt(np.mean(steady**2))),max_m=float(max(errors)),
            final_m=float(errors[-1]),delta_v_m_s=float(total_dv),fuel_kg=float(500-mass),burn_count=issued,
            modes=dict(Counter(decisions)),planner_cpu_s=cpu_planning,navigation_cpu_s=cpu_navigation,
            total_cpu_s=time.process_time()-started,wall_s=time.perf_counter()-wall,
            max_planner_cpu_s=max_plan_cpu,max_online_force_calls=max_calls,force_budget_violations=budget_violations,
            fixes_accepted=filter_.accepted if navigation=='filter' else None,fixes_rejected=filter_.rejected if navigation=='filter' else None,
            bias_clips=filter_.bias_clips if navigation=='filter' else None,
            median_nis=None if not nis else float(np.median(nis)),mean_nees=None if not nees else float(np.mean(nees)),
            navigation_rms_m=float(np.sqrt(np.mean([s['navigation_error_m']**2 for s in samples]))),
            settings=asdict(settings),filter_settings=asdict(filter_.settings),samples=samples,commands=records)
        return row


def select(training,method):
    if any(r['scenario']['name'] not in {s.name for s in TRAINING} for r in training):
        raise ValueError('Selection accepts training scenarios only')
    candidates=[]
    for name,(kind,_) in configurations().items():
        if kind!=method:continue
        rows=[r for r in training if r['configuration']==name]
        if len(rows)!=len(TRAINING):raise ValueError('Incomplete training set')
        feasible=(max(r['steady_rms_m'] for r in rows)<=500.
                  and np.mean([r['steady_rms_m'] for r in rows])<=300.
                  and max(r['max_m'] for r in rows)<=5000.
                  and sum(r['force_budget_violations'] for r in rows)==0)
        candidates.append(dict(configuration=name,feasible=bool(feasible),
            mean_delta_v_m_s=float(np.mean([r['delta_v_m_s'] for r in rows])),
            mean_steady_rms_m=float(np.mean([r['steady_rms_m'] for r in rows])),
            worst_steady_rms_m=float(max(r['steady_rms_m'] for r in rows))))
    eligible=[r for r in candidates if r['feasible']]
    winner=min(eligible,key=lambda r:r['mean_delta_v_m_s']) if eligible else min(candidates,key=lambda r:r['worst_steady_rms_m'])
    return dict(selected=winner['configuration'],meets_training_constraints=bool(eligible),candidates=candidates)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--physics-dir',type=Path,required=True);p.add_argument('--ephemeris',type=Path,required=True)
    p.add_argument('--gm-kernel',type=Path,required=True);p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--phase',choices=['pilot','train','evaluate','month'],required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    protocol=dict(classification='synthetic direct-state navigation/control benchmark; not flight hardware or raw-sensor validation',
        training=[asdict(s) for s in TRAINING],evaluation=[asdict(s) for s in EVALUATION],month=asdict(LONG),
        configurations={name:dict(method=m,settings=asdict(s)) for name,(m,s) in configurations().items()},
        selection='Training only: mean steady RMS<=300m, worst scenario<=500m, max error<=5km and <=5000 force evaluations per decision; then minimum mean delta-v. If none feasible, flag failure and select lowest worst RMS for diagnosis.',
        scoring='Six-hour epochs, steady RMS excludes first24h; report full RMS and maxima too',
        pairing='Same independent Cartesian plant, reference, timestamp-keyed random draws and actuator limits across variants',
        filter=asdict(FilterSettings()))
    protocol_path=args.output/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text())!=protocol:raise RuntimeError('Protocol changed: use a new experiment directory')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    model,inputs=make_model(args.physics_dir,args.ephemeris,args.gm_kernel,args.native_library,32*DAY)
    experiment=Experiment(model,inputs)
    source={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ('control_tuning.py','adaptive_control.py','jacobi_control.py','jacobi_control_demo.py')}
    output=args.output/(args.phase+'.json')
    report=dict(protocol_sha256=hashlib.sha256(protocol_path.read_bytes()).hexdigest(),source_sha256=source,inputs=inputs,results=[])
    if args.phase=='pilot':
        cases=[Scenario('pilot_not_scored',9901,3,1e7,45,outlier=True)]
        jobs=[('raw','mpc_original_weights','jacobi','raw',False),('filtered_mpc','mpc_original_weights','jacobi','filter',True),('pd','pd_12h','jacobi','filter',True),('target','target_12h','jacobi','filter',True)]
    elif args.phase=='train':
        cases=TRAINING;jobs=[(name,name,'jacobi','filter',True) for name in configurations()]
    else:
        trained=json.loads((args.output/'train.json').read_text())
        selections=trained['selection']
        if trained['protocol_sha256']!=report['protocol_sha256']:raise ValueError('Training protocol mismatch')
        if trained['source_sha256']!=source:raise ValueError('Implementation changed after training; start a new experiment')
        mpc,pd,target=selections['mpc']['selected'],selections['pd']['selected'],selections['target']['selected']
        cases=EVALUATION if args.phase=='evaluate' else [LONG]
        jobs=[('original','mpc_original_weights','jacobi','raw',False),('tuned_mpc',mpc,'jacobi','filter',True),
              ('tuned_pd',pd,'jacobi','filter',True),('tuned_target',target,'jacobi','filter',True)]
        if args.phase=='evaluate':jobs.extend([('cartesian_mpc',mpc,'cartesian','filter',True),('no_bias_feedforward',mpc,'jacobi','filter',False)])
        report['frozen_selection']=selections
    for case in cases:
        for label,config,coordinates,navigation,bias in jobs:
            print('starting',args.phase,case.name,label,flush=True)
            row=experiment.run(case,configuration=config,coordinates=coordinates,navigation=navigation,bias=bias)
            row['label']=label;report['results'].append(row)
            output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(case.name,label,'steady RMS',round(row['steady_rms_m'],2),'DV',round(row['delta_v_m_s'],5),
                  'burns',row['burn_count'],'rejected',row['fixes_rejected'],'CPU',round(row['total_cpu_s'],2),flush=True)
    if args.phase=='train':
        report['selection']={m:select(report['results'],m) for m in ('mpc','pd','target')}
        output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print('selection',report['selection'],flush=True)


if __name__=='__main__':main()
