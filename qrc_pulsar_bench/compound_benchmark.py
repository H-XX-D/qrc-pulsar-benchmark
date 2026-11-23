"""Blind concurrent/sequential fault simulation with causal input hygiene.

Fault schedules control only the simulated plant, sensor and link. The flight
filter, MPC and event recorder see measurements, issued commands and link status.
"""
from dataclasses import asdict,replace
import hashlib
import time
import traceback
import numpy as np
from scipy.spatial.transform import Rotation
from .adaptive_control import MoonCartesianFlightModel
from .anomaly_benchmark import Anomaly,score_recovery
from .compound_faults import FaultCase
from .event_journal import EventJournal
from .maneuvers import FiniteBurn
from .navigation_ingress import NavigationIngress
from .navigation_recovery import ManeuverAwareOrbitFilter
from .resilience_benchmark import ResilienceExperiment,DAY
from .routine_supervisor import RoutineMPC


class PlantRollout:
    def __init__(self,parts,propulsive_dv):
        self.parts=parts;self.propulsive_dv=propulsive_dv
        self.start_s=parts[0].start_s;self.stop_s=parts[-1].stop_s
        self.completed=all(p.completed for p in parts)
        self.min_clearance_m=min(p.min_clearance_m for p in parts)

    def at(self,t):
        for part in self.parts:
            if part.start_s-1e-7<=t<=part.stop_s+1e-7:return part.at(t)
        raise ValueError('Time outside plant history')


def propagate_plant(model,state,start,stop,mass,command,leak_force,leak_flow):
    """Combine overlapping propulsion vectors AND their propellant flows.

    Split exactly when the command ends. Uncommanded thrust continues through
    the rest of the interval, so it cannot be hidden as a mass-free acceleration.
    """
    boundaries=sorted({start,stop,*([command.stop_s] if command is not None and start<command.stop_s<stop else [])})
    parts=[];dv=0.
    for left,right in zip(boundaries[:-1],boundaries[1:]):
        force=np.array(leak_force,dtype=float);flow=float(leak_flow)
        if command is not None and command.start_s<=left<command.stop_s:
            force+=command.thrust_n*np.asarray(command.direction_icrf);flow+=command.mass_flow_kg_s
        norm=float(np.linalg.norm(force));burn=None
        if norm>0 or flow>0:
            direction=force/norm if norm else np.array([1.,0,0])
            burn=FiniteBurn(left,right,norm,tuple(direction),mass,flow,left,'simulated combined actual propulsion')
        part=model.propagate(state,left,right,mass,burn);parts.append(part)
        final=part.at(part.stop_s)
        if flow>0:dv+=norm/flow*np.log(mass/final[6])
        elif norm>0:dv+=norm/mass*(part.stop_s-left)
        state=final[:6];mass=float(final[6])
        if not part.completed:break
    return PlantRollout(parts,float(dv))


class CompoundExperiment(ResilienceExperiment):
    def run(self,scenario,case:FaultCase,*,library=None,coast_after_event=False,journal_path=None):
        if case.onset_s<12*3600 or case.end_s>scenario.days*DAY:
            raise ValueError('Fault schedule outside experiment')
        initial,model,plan,inputs=self.prepare(scenario)
        if plan.min_clearance_m<50000:raise ValueError('Reference violates50km clearance floor')
        controller=RoutineMPC(model,plan,library=library,bits=32)
        actuator=controller.actuator;h=controller.settings.interval_s
        rng=np.random.default_rng(scenario.seed)
        actual=initial+np.r_[rng.normal(size=3)*300,rng.normal(size=3)*.01]
        bias=rng.normal(size=3);bias*=scenario.bias_m_s2/np.linalg.norm(bias)
        # Separate, frozen exogenous streams. Flight code never receives these.
        fault_rng=np.random.default_rng(scenario.seed+int.from_bytes(hashlib.sha256(case.name.encode()).digest()[:4],'little'))
        directions=fault_rng.normal(size=(len(case.events),3))
        if len(directions):directions/=np.linalg.norm(directions,axis=1)[:,None]
        events=list(zip(case.events,directions))
        nav=ManeuverAwareOrbitFilter(model,initial,np.diag(np.r_[[500.]*3,[.03]*3]**2),500.)
        ingress=NavigationIngress(nav);covariance=np.diag(np.r_[[30.]*3,[.002]*3]**2)
        journal=EventJournal(journal_path);received={};delivery_times={};previous_health=None;previous_link=True
        telemetry_errors=[];dropped_events=0;storage_failed=False
        ledger=[];history=None;last_packet=None;mass=500.;records=[];samples=[];dense=[]
        collision=False;failure=None;net_dv=0.;minimum_clearance=np.inf;start=time.process_time()

        def point(t,state):
            error=state[:6]-plan.at(t)[:6]
            return dict(time_s=float(t),position_error_m=float(np.linalg.norm(error[:3])),velocity_error_m_s=float(np.linalg.norm(error[3:])))

        def publish(t,event,details):
            nonlocal dropped_events,storage_failed
            if storage_failed:
                dropped_events+=1;return
            try:journal.publish(t,event,details)
            except (OSError,OverflowError) as error:
                # After an uncertain disk write, do not append more records to
                # a potentially incomplete journal. Guidance remains live.
                if isinstance(error,OSError):storage_failed=True
                dropped_events+=1
                if len(telemetry_errors)<32:telemetry_errors.append(dict(time_s=float(t),message=str(error)))

        def deliver(receiver,t):
            nonlocal storage_failed
            if storage_failed:return
            try:journal.deliver(receiver)
            except OSError as error:
                storage_failed=True
                if len(telemetry_errors)<32:telemetry_errors.append(dict(time_s=float(t),message=str(error)))

        dense.append(point(0.,actual))
        for t in np.arange(0,scenario.days*DAY+1,h):
            stage='sensor'
            try:
                noise=rng.normal(size=6);magnitude=rng.normal()*scenario.thrust_sigma
                pointing=rng.normal(size=3)*scenario.pointing_sigma_rad
                for fault,direction in events:
                    if fault.kind=='impulse' and t==fault.start_s:actual[3:]+=fault.value*direction
                if dense[-1]['time_s']==t:dense[-1]=point(t,actual)
                active=[(f,d) for f,d in events if f.active(t)]
                ground_up=not any(f.kind=='ground_blackout' for f,d in active)
                if ground_up!=previous_link:
                    publish(t,'ground_link_restored' if ground_up else 'ground_link_lost',{})
                    previous_link=ground_up
                observation=None;stamp=max(0.,t-60.)
                if not any(f.kind=='sensor_dropout' for f,d in active):
                    measured=actual.copy() if stamp==0 else history.at(stamp)[:6]
                    sensor_events=[(f,d) for f,d in events if f.active(stamp)]
                    noise_gain=np.prod([f.value for f,d in sensor_events if f.kind=='sensor_noise'])
                    fix=measured+noise*np.r_[[30.]*3,[.002]*3]*noise_gain
                    reported_stamp=stamp
                    for fault,direction in sensor_events:
                        if fault.kind=='sensor_bias':fix+=np.r_[direction*fault.value,direction*fault.second_value]
                        elif fault.kind=='sensor_clock':reported_stamp+=fault.value
                        elif fault.kind=='sensor_nan':fix[3]=np.nan
                    if any(f.kind=='sensor_replay' for f,d in sensor_events) and last_packet is not None:
                        fix,reported_stamp=last_packet[0].copy(),last_packet[1]
                    last_packet=(fix.copy(),reported_stamp)
                    stage='navigation'
                    observation=ingress.observe(fix,covariance,reported_stamp,t,ledger)
                    if observation['input_error']:
                        publish(t,'sensor_packet_rejected',dict(reason=observation['input_error']))
                stage='navigation'
                estimate=ingress.forecast(t,ledger,as_of_s=t)
                ledger=[b for b in ledger if b.stop_s>nav.time_s]
                error=actual-estimate['state'][:6];p=estimate['covariance']
                health=(estimate['navigation_mode'],estimate['navigation_trusted'])
                if health!=previous_health:
                    publish(t,'navigation_status',dict(mode=health[0],trusted=health[1],
                        observation_age_s=float(t-estimate['last_observation_s'])))
                    previous_health=health
                if observation is not None and observation.get('reacquired'):
                    publish(t,'navigation_reacquired',dict(nav.events[-1]))
                samples.append(dict(**point(t,actual),navigation_error_m=float(np.linalg.norm(error[:3])),
                    navigation_velocity_error_m_s=float(np.linalg.norm(error[3:])),
                    navigation_trusted=estimate['navigation_trusted'],navigation_mode=estimate['navigation_mode'],
                    accepted=None if observation is None else observation['accepted'],nis=None if observation is None else observation['nis'],
                    input_error=None if observation is None else observation['input_error'],age_s=float(t-estimate['last_observation_s']),
                    position_3sigma_m=float(3*np.sqrt(np.linalg.eigvalsh(p[:3,:3])[-1])),ground_link_up=ground_up))

                def receiver(record):
                    if record['sequence'] not in received:
                        received[record['sequence']]=record;delivery_times[record['sequence']]=float(t)
                    return True

                if t>=scenario.days*DAY:
                    if ground_up:deliver(receiver,t)
                    break
                stage='controller'
                if coast_after_event and t>=case.onset_s:
                    dv=np.zeros(3);status=256;mode='comparison_coast'
                else:
                    dv=controller.command(float(t),estimate,estimate['mass']);status=controller.last_status;mode=controller.mode
                requested=actuator.command(dv,t,estimate['mass'],interval_s=h) if status==0 else None
                if not records or status!=records[-1]['status']:
                    publish(t,'guidance_status',dict(status=int(status),mode=mode))
                if requested is not None:
                    ledger.append(requested)
                    publish(t,'command_issued',dict(dv_m_s=dv.tolist(),estimated_mass_kg=float(estimate['mass'])))
                if ground_up:deliver(receiver,t)
                executed=None;gain=float(np.prod([f.value for f,d in active if f.kind=='thrust_gain']))
                if requested is not None and gain>0:
                    direction=Rotation.from_rotvec(pointing).apply(requested.direction_icrf)
                    for fault,axis in active:
                        if fault.kind=='pointing':direction=Rotation.from_rotvec(axis*fault.value).apply(direction)
                    executed=replace(requested,mass_start_kg=mass,thrust_n=requested.thrust_n*gain*(1+magnitude),
                        mass_flow_kg_s=requested.mass_flow_kg_s*gain,direction_icrf=tuple(direction))
                force_events=[(f,d) for f,d in active if f.kind in ('acceleration','acceleration_ramp')]
                def acceleration(r,v,epoch):
                    extra=bias.copy()
                    for fault,direction in force_events:
                        scale=1. if fault.kind=='acceleration' else np.clip((epoch-fault.start_s)/(fault.end_s-fault.start_s),0,1)
                        extra+=direction*fault.value*scale
                    return model.acceleration(r,v,epoch)+extra
                leak_force=sum((d*f.value for f,d in active if f.kind=='uncommanded_thrust'),np.zeros(3))
                leak_flow=sum(f.value/actuator.exhaust_velocity for f,d in active if f.kind=='uncommanded_thrust')
                plant=MoonCartesianFlightModel(acceleration,model.inner,moon_radius_m=model.moon_radius_m,
                    earth_radius_m=model.earth_radius_m,max_step_s=150.,rtol=2e-13)
                stage='plant'
                rollout=propagate_plant(plant,actual,t,t+h,mass,executed,leak_force,leak_flow)
                minimum_clearance=min(minimum_clearance,rollout.min_clearance_m);net_dv+=rollout.propulsive_dv
                records.append(dict(time_s=float(t),dv=dv.tolist(),burn_s=0. if requested is None else requested.stop_s-requested.start_s,
                    actual_burn_s=0. if executed is None else min(executed.stop_s,rollout.stop_s)-executed.start_s,
                    uncommanded_thrust_n=float(np.linalg.norm(leak_force)),status=int(status),mode=mode))
                for epoch in np.arange(t+300,min(t+h,rollout.stop_s)+1,300):dense.append(point(epoch,rollout.at(epoch)))
                final=rollout.at(rollout.stop_s);actual=final[:6].copy();mass=float(final[6]);history=rollout
                if not rollout.completed:
                    collision=True
                    if dense[-1]['time_s']!=rollout.stop_s:dense.append(point(rollout.stop_s,actual))
                    break
            except (ValueError,RuntimeError,np.linalg.LinAlgError) as exc:
                failure=dict(type=type(exc).__name__,message=str(exc),stage=stage,time_s=float(t),traceback=traceback.format_exc())
                records.append(dict(time_s=float(t),dv=[0.,0.,0.],burn_s=0.,actual_burn_s=0.,
                    uncommanded_thrust_n=0.,status=512,mode='simulation_exception'))
                break
        completed=bool(not collision and failure is None and dense[-1]['time_s']==scenario.days*DAY)
        envelope=Anomaly(case.name,case.onset_s,case.end_s)
        scored=(score_recovery(dense,envelope,completed=completed) if dense[-1]['time_s']>=case.onset_s else
                dict(response='not_recovered',settled=False,settling_after_onset_h=None,settling_after_end_h=None,
                    post_event_rms_m=None,post_event_peak_m=None,final_error_m=dense[-1]['position_error_m'],
                    final_velocity_error_m_s=dense[-1]['velocity_error_m_s']))
        issued=float(sum(np.linalg.norm(c['dv']) for c in records if c['burn_s']>0))
        statuses={str(s):sum(c['status']==s for c in records) for s in {c['status'] for c in records}}
        # Distinguish observable navigation restoration from nominal-path return.
        eligible=[s for s in samples if s['time_s']>=case.end_s]
        bad=lambda s:not s['navigation_trusted'] or s['navigation_error_m']>200 or s['navigation_velocity_error_m_s']>.02
        last_bad=max((i for i,s in enumerate(eligible) if bad(s)),default=-1)
        tail=eligible[last_bad+1:]
        navigation_settled=bool(completed and tail and tail[-1]['time_s']-tail[0]['time_s']>=21600)
        max_latency=max((delivery_times[seq]-record['time_s'] for seq,record in received.items()),default=0.)
        row=dict(scenario=asdict(scenario),case=asdict(case),fault_directions=directions.tolist(),
            arithmetic='float64' if library is None else '32',control='coast_after_event' if coast_after_event else 'routine_mpc',
            initial_state=initial.tolist(),model=inputs,nominal_min_clearance_m=float(plan.min_clearance_m),
            metrics=dict(**scored,completed=completed,collision=collision,min_clearance_m=float(minimum_clearance) if np.isfinite(minimum_clearance) else None,
                issued_dv_m_s=issued,net_propulsive_dv_m_s=float(net_dv),propellant_kg=500.-mass,
                remaining_propellant_kg=mass-actuator.dry_mass_kg),
            samples=samples,dense_samples=dense,commands=records,navigation_events=list(nav.events),
            telemetry=dict(events=journal.sequence,delivered=len(received),pending=len(journal.pending),
                dropped_events=dropped_events,storage_failed=storage_failed,errors=telemetry_errors,
                maximum_pending=journal.maximum_pending,maximum_delivery_latency_s=float(max_latency),
                journal_path=None if journal.path is None else str(journal.path)),cpu_s=time.process_time()-start)
        row['metrics'].update(burns=sum(bool(c['burn_s']>0) for c in records),status_counts=statuses,
            fixes_accepted=nav.accepted,fixes_rejected=nav.rejected,invalid_packets=ingress.invalid_packets,
            navigation_reacquisitions=nav.reacquisitions,navigation_settled=navigation_settled,
            final_navigation_error_m=samples[-1]['navigation_error_m'] if samples else None,
            final_navigation_velocity_error_m_s=samples[-1]['navigation_velocity_error_m_s'] if samples else None)
        if failure is not None:row['failure']=failure
        return row
