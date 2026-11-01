"""Blind fault injection around frozen MPC; truth is used only by the scorer.

Anomaly parameters never enter the controller, navigation filter or reference.
Impulses are plant-only velocity jumps; delayed fixes at the jump epoch still
describe the pre-jump state. Recovery requires a sustained position AND velocity
corridor, not merely crossing the reference once.
"""
from dataclasses import asdict,dataclass,replace
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .adaptive_control import CausalOrbitFilter,MoonCartesianFlightModel
from .resilience_benchmark import ResilienceExperiment,Scenario,DAY
from .routine_supervisor import RoutineMPC


@dataclass(frozen=True)
class Anomaly:
    name: str
    onset_s: float
    end_s: float
    kick_m_s: float = 0.
    acceleration_m_s2: float = 0.
    thrust_gain: float = 1.
    sensor_position_bias_m: float = 0.
    sensor_velocity_bias_m_s: float = 0.
    outage_s: float = 0.
    persistent: bool = False

    def __post_init__(self):
        values=[self.onset_s,self.end_s,self.kick_m_s,self.acceleration_m_s2,
                self.thrust_gain,self.sensor_position_bias_m,self.sensor_velocity_bias_m_s,self.outage_s]
        if not np.all(np.isfinite(values)) or min(values)<0 or self.end_s<self.onset_s:
            raise ValueError('Invalid anomaly parameters')
        if self.thrust_gain>1 or self.outage_s>self.end_s-self.onset_s:
            raise ValueError('Invalid thrust/outage bounds')
        if self.onset_s%1800 or self.end_s%1800 or self.outage_s%1800:
            raise ValueError('Fault boundaries must align with the control grid')

    def active(self,t):return self.onset_s<=t<self.end_s


def score_recovery(points,anomaly,*,completed,position_m=100.,velocity_m_s=.02,hold_s=21600.):
    """Last excursion must be followed by >=6h in bounds through the run end.

For a persistent fault, success means regulation while the fault remains active.
For a transient fault, settling is scored only after it ends. Event peak and RMS
include the whole response, including the disturbance interval.
"""
    after=[p for p in points if p['time_s']>=anomaly.onset_s]
    eligible=anomaly.onset_s if anomaly.persistent else anomaly.end_s
    tail=[p for p in after if p['time_s']>=eligible]
    outside=lambda p:p['position_error_m']>position_m or p['velocity_error_m_s']>velocity_m_s
    last_out=max((i for i,p in enumerate(tail) if outside(p)),default=-1)
    settling=tail[last_out+1]['time_s'] if tail and last_out+1<len(tail) else None
    settled=bool(completed and settling is not None and tail[-1]['time_s']-settling>=hold_s)
    stayed_inside=not any(outside(p) for p in after)
    return dict(response=('contained' if stayed_inside else 'recovered') if settled else 'not_recovered',
        settled=settled,settling_after_onset_h=(settling-anomaly.onset_s)/3600 if settled else None,
        settling_after_end_h=(settling-eligible)/3600 if settled else None,
        post_event_rms_m=float(np.sqrt(np.mean([p['position_error_m']**2 for p in after]))),
        post_event_peak_m=max(p['position_error_m'] for p in after),
        final_error_m=points[-1]['position_error_m'],final_velocity_error_m_s=points[-1]['velocity_error_m_s'])


class AnomalyExperiment(ResilienceExperiment):
    def run_anomaly(self,scenario,anomaly,*,library=None,coast_after_event=False,filter_factory=None):
        if anomaly.end_s>scenario.days*DAY or anomaly.onset_s<DAY/2:
            raise ValueError('Fault outside the declared experiment')
        initial,model,plan,inputs=self.prepare(scenario)
        if plan.min_clearance_m<50000:
            raise ValueError('Reference violates the declared 50 km lunar clearance floor')
        controller=RoutineMPC(model,plan,library=library,bits=32)
        actuator=controller.actuator;h=controller.settings.interval_s
        rng=np.random.default_rng(scenario.seed)
        actual=initial+np.r_[rng.normal(size=3)*300,rng.normal(size=3)*.01]
        bias=rng.normal(size=3);bias*=scenario.bias_m_s2/np.linalg.norm(bias)
        fault_rng=np.random.default_rng(scenario.seed+100000)
        direction=fault_rng.normal(size=3);direction/=np.linalg.norm(direction)
        nav=(CausalOrbitFilter if filter_factory is None else filter_factory)(
            model,initial,np.diag(np.r_[[500.]*3,[.03]*3]**2),500.)
        covariance=np.diag(np.r_[[30.]*3,[.002]*3]**2)
        ledger=[];history=None;mass=500.;records=[];samples=[];dense=[]
        collision=False;actual_dv=0.;minimum_clearance=np.inf;start=time.process_time()

        def record_point(t,state):
            error=state[:6]-plan.at(t)[:6]
            return dict(time_s=float(t),position_error_m=float(np.linalg.norm(error[:3])),
                        velocity_error_m_s=float(np.linalg.norm(error[3:])))

        for t in np.arange(0,scenario.days*DAY+1,h):
            # Draw the same exogenous noise even if a measurement or burn is absent.
            noise=rng.normal(size=6);magnitude=rng.normal()*scenario.thrust_sigma
            pointing=rng.normal(size=3)*scenario.pointing_sigma_rad
            if t==anomaly.onset_s:
                actual=actual.copy();actual[3:]+=anomaly.kick_m_s*direction
                if dense:dense[-1]=record_point(t,actual)
            stamp=max(0,t-60.);observation=None
            if not anomaly.onset_s<=t<anomaly.onset_s+anomaly.outage_s:
                measured=actual if stamp==0 else history.at(stamp)[:6]
                fix=measured+noise*np.r_[[30.]*3,[.002]*3]
                # Sensor bias is tied to measurement time, not delivery time.
                if anomaly.active(stamp):
                    fix+=np.r_[direction*anomaly.sensor_position_bias_m,direction*anomaly.sensor_velocity_bias_m_s]
                observation=nav.observe(fix,covariance,stamp,t,ledger)
            estimate=nav.forecast(t,ledger,as_of_s=t)
            ledger=[b for b in ledger if b.stop_s>nav.time_s]
            point=record_point(t,actual)
            samples.append(dict(**point,navigation_error_m=float(np.linalg.norm(actual[:3]-estimate['state'][:3])),
                age_s=float(t-estimate['last_observation_s']),accepted=None if observation is None else observation['accepted'],
                nis=None if observation is None else observation['nis']))
            if filter_factory is not None:
                # Truth-based diagnostics stay in the scoring harness only.
                error=actual-estimate['state'][:6]
                p=estimate['covariance'];scales=np.sqrt(np.diag(p))
                samples[-1].update(navigation_velocity_error_m_s=float(np.linalg.norm(error[3:])),
                    navigation_trusted=estimate.get('navigation_trusted',True),
                    navigation_mode=estimate.get('navigation_mode','tracking'),
                    candidate_confirmations=estimate.get('candidate_confirmations',0),
                    reacquired=False if observation is None else observation.get('reacquired',False),
                    position_3sigma_m=float(3*np.sqrt(np.linalg.eigvalsh(p[:3,:3])[-1])),
                    velocity_3sigma_m_s=float(3*np.sqrt(np.linalg.eigvalsh(p[3:6,3:6])[-1])),
                    covariance_correlation_min_eigenvalue=float(np.linalg.eigvalsh(p/np.outer(scales,scales))[0]))
            if t==0:dense.append(point)
            if t>=scenario.days*DAY:break
            if coast_after_event and t>=anomaly.onset_s:
                dv=np.zeros(3);status=256;mode='comparison_coast'
            else:
                # No truth mass, fault identity, injection times or simulator state.
                dv=controller.command(float(t),estimate,estimate['mass'])
                status=controller.last_status;mode=controller.mode
            requested=actuator.command(dv,t,estimate['mass'],interval_s=h) if status==0 else None
            executed=None;gain=anomaly.thrust_gain if anomaly.active(t) else 1.
            if requested is not None:
                ledger.append(requested)
                if gain>0:
                    executed=replace(requested,mass_start_kg=mass,thrust_n=requested.thrust_n*gain*(1+magnitude),
                        mass_flow_kg_s=requested.mass_flow_kg_s*gain,
                        direction_icrf=tuple(Rotation.from_rotvec(pointing).apply(requested.direction_icrf)))
                    duration=executed.stop_s-executed.start_s
                    actual_dv+=executed.thrust_n/executed.mass_flow_kg_s*np.log(mass/(mass-executed.mass_flow_kg_s*duration))
            # Freeze forcing on each integration interval: discontinuities are
            # exact boundaries, not events smeared by an adaptive ODE step.
            extra=direction*anomaly.acceleration_m_s2 if anomaly.active(t) else np.zeros(3)
            plant=MoonCartesianFlightModel(lambda r,v,epoch:model.acceleration(r,v,epoch)+bias+extra,
                model.inner,moon_radius_m=model.moon_radius_m,earth_radius_m=model.earth_radius_m,max_step_s=150.,rtol=2e-13)
            rollout=plant.propagate(actual,t,t+h,mass,executed)
            minimum_clearance=min(minimum_clearance,rollout.min_clearance_m)
            records.append(dict(time_s=float(t),dv=dv.tolist(),burn_s=0. if requested is None else requested.stop_s-requested.start_s,
                actual_burn_s=0. if executed is None else executed.stop_s-executed.start_s,status=status,mode=mode))
            for stamp in np.arange(t+300,min(t+h,rollout.stop_s)+1,300):
                dense.append(record_point(stamp,rollout.at(stamp)))
            actual,mass=rollout.at(rollout.stop_s)[:6],float(rollout.at(rollout.stop_s)[6]);history=rollout
            if not rollout.completed:
                collision=True
                if dense[-1]['time_s']!=rollout.stop_s:dense.append(record_point(rollout.stop_s,actual))
                break
        completed=bool(not collision and dense[-1]['time_s']==scenario.days*DAY)
        scored=score_recovery(dense,anomaly,completed=completed)
        issued=sum(np.linalg.norm(c['dv']) for c in records if c['burn_s']>0)
        statuses={str(s):sum(c['status']==s for c in records) for s in {c['status'] for c in records}}
        return dict(scenario=asdict(scenario),anomaly=asdict(anomaly),direction_icrf=direction.tolist(),
            arithmetic='float64' if library is None else '32',control='coast_after_event' if coast_after_event else 'routine_mpc',
            initial_state=initial.tolist(),model=inputs,nominal_min_clearance_m=plan.min_clearance_m,
            metrics=dict(**scored,completed=completed,collision=collision,min_clearance_m=float(minimum_clearance),
                issued_dv_m_s=float(issued),delivered_dv_m_s=float(actual_dv),propellant_kg=500.-mass,
                burns=sum(bool(c['burn_s']>0) for c in records),actual_burns=sum(bool(c['actual_burn_s']>0) for c in records),
                status_counts=statuses,fixes_rejected=nav.rejected,fixes_accepted=nav.accepted,
                navigation_reacquisitions=getattr(nav,'reacquisitions',0),
                first_recovery_required_h=next((c['time_s']/3600 for c in records if c['status']==64),None)),
            cpu_s=time.process_time()-start,samples=samples,dense_samples=dense,commands=records,
            navigation_events=list(getattr(nav,'events',[])))
