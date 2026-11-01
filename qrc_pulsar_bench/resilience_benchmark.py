"""Synthetic outage/endurance tests using only a definitive ARTEMIS initial state."""
from dataclasses import dataclass,asdict,replace
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from .artemis_replay import make_model,DAY
from .adaptive_control import CausalOrbitFilter,MoonCartesianFlightModel
from .jacobi_control import Actuator
from .mpc_strategies import StrategyMPC,strategies
from .resilient_mpc import GuardedMPC,profiles


@dataclass(frozen=True)
class Scenario:
    name: str
    probe: int
    utc: str
    days: float
    seed: int
    split: str
    outages: tuple = ()             # elapsed hours, never supplied to guidance
    pulses: tuple = ((24.,36.,3e-7),)
    bias_m_s2: float = 8e-8
    thrust_sigma: float = .03
    pointing_sigma_rad: float = .005

    def __post_init__(self):
        if self.probe not in (1,2) or not np.isfinite(self.days) or self.days<=.5:
            raise ValueError('Invalid scenario')
        if not np.isclose(self.days*48,round(self.days*48)):raise ValueError('Duration must align with control grid')
        for a,b in self.outages:
            if not 0<=a<b<=self.days*24:raise ValueError('Outage outside simulation')
        for a,b,magnitude in self.pulses:
            if not 0<=a<b<=self.days*24 or not np.isfinite(magnitude):raise ValueError('Pulse outside simulation')


class ResilienceExperiment:
    def __init__(self,data,kernels,physics,native):
        self.paths=tuple(Path(p) for p in (data,kernels,physics,native));self.cache={}

    def prepare(self,scenario):
        key=(scenario.probe,scenario.utc,scenario.days)
        if key not in self.cache:
            import spiceypy as sp
            data,kernels,physics,native=self.paths
            sp.kclear()
            for name in ('naif0012.tls','de440s.bsp','gm_de440.tpc'):sp.furnsh(str(kernels/name))
            spk=data/f'ARTEMIS_P{scenario.probe}_DEF_2009022_2012022.bsp';sp.furnsh(str(spk))
            epoch=sp.str2et(scenario.utc)
            # Exactly one spacecraft state is read. Future planetary ephemeris
            # supplies prescribed forces; future spacecraft data is not read.
            initial=np.asarray(sp.spkezr(str(-650-scenario.probe),epoch,'J2000','NONE','301')[0])*1000.
            model,inputs=make_model(2451545.+epoch/DAY,scenario.days*DAY,physics,kernels/'de440s.bsp',kernels/'gm_de440.tpc',native)
            plan=model.propagate(initial,0,scenario.days*DAY,500.)
            if not plan.completed:raise RuntimeError('Nominal reference does not cover mission')
            self.cache[key]=(initial,model,plan,inputs)
            if len(self.cache)>2:del self.cache[next(iter(self.cache))]
        return self.cache[key]

    def run(self,scenario,profile,*,library=None,guidance_factory=None):
        initial,model,plan,inputs=self.prepare(scenario);spec=profiles()[profile]
        kwargs=dict(library=library,bits=32)
        controller=(GuardedMPC(model,plan,spec['guard'],**kwargs) if spec['guard'] else
                    StrategyMPC(model,plan,replace(strategies()['sparse_05'],use_bias=spec['bias']),**kwargs))
        if guidance_factory is not None:controller=guidance_factory(model,plan,**kwargs)
        actuator=controller.actuator;h=controller.settings.interval_s
        rng=np.random.default_rng(scenario.seed)
        actual=initial+np.r_[rng.normal(size=3)*300,rng.normal(size=3)*.01]
        bias=rng.normal(size=3);bias*=scenario.bias_m_s2/np.linalg.norm(bias)
        def acceleration(r,v,t):
            tangent=np.cross(np.cross(r,v),r);tangent/=np.linalg.norm(tangent)
            pulse=sum(magnitude for a,b,magnitude in scenario.pulses if a*3600<=t<b*3600)
            return model.acceleration(r,v,t)+bias+pulse*tangent
        plant=MoonCartesianFlightModel(acceleration,model.inner,moon_radius_m=model.moon_radius_m,max_step_s=150.,rtol=2e-13)
        nav=CausalOrbitFilter(model,initial,np.diag(np.r_[[500.]*3,[.03]*3]**2),500.,settings=spec['filter'])
        covariance=np.diag(np.r_[[30.]*3,[.002]*3]**2)
        commands=[];history=None;mass=500.;dv_total=0.;burns=0;online=0.;collision=False
        records=[];samples=[];dense=[];modes={};statuses={};max_caches=[0,0,0]
        start=time.process_time()
        for t in np.arange(0,scenario.days*DAY+1,h):
            noise=rng.normal(size=6);magnitude=rng.normal()*scenario.thrust_sigma;pointing=rng.normal(size=3)*scenario.pointing_sigma_rad
            tic=time.process_time();stamp=max(0,t-60.);observation=None
            if not any(a*3600<=t<b*3600 for a,b in scenario.outages):
                measured=actual if stamp==0 else history.at(stamp)[:6]
                fix=measured+noise*np.r_[[30.]*3,[.002]*3]
                if scenario.outages and t==2*DAY:fix+=np.r_[10000.,-8000.,5000.,.1,-.1,.05]
                observation=nav.observe(fix,covariance,stamp,t,commands)
            estimate=nav.forecast(t,commands,as_of_s=t)
            # A command wholly incorporated into the filter is no longer needed
            # by the onboard ledger. Full requested-burn records remain below.
            commands=[b for b in commands if b.stop_s>nav.time_s]
            error=float(np.linalg.norm(actual[:3]-plan.at(t)[:3]))
            sigma=float(np.sqrt(np.linalg.eigvalsh(estimate['covariance'][:3,:3])[-1]))
            samples.append(dict(time_s=float(t),error_m=error,navigation_error_m=float(np.linalg.norm(actual[:3]-estimate['state'][:3])),
                position_sigma_m=sigma,age_s=float(t-estimate['last_observation_s']),bias_estimate=estimate['state'][6:].tolist(),
                accepted=None if observation is None else observation['accepted'],nis=None if observation is None else observation['nis']))
            if t==0:dense.append(dict(time_s=0.,error_m=error))
            if t>=scenario.days*DAY:break
            valid=True if spec['guard'] else t-estimate['last_observation_s']<=3600.
            dv=controller.command(float(t),estimate,mass,valid=valid)
            status=controller.last_status;statuses[str(status)]=statuses.get(str(status),0)+1
            mode=getattr(controller,'mode','fresh' if valid else 'inhibited')
            modes[mode]=modes.get(mode,0)+1
            burn=actuator.command(dv,t,mass,interval_s=h) if status==0 else None
            online+=time.process_time()-tic
            max_caches=[max(max_caches[0],len(controller.cache)),max(max_caches[1],len(controller.affine_cache)),max(max_caches[2],len(controller.diagnostics))]
            executed=None
            if burn is not None:
                commands.append(burn);burns+=1;spent=burn.mass_flow_kg_s*(burn.stop_s-burn.start_s)
                dv_total+=actuator.exhaust_velocity*np.log(mass/(mass-spent))
                executed=replace(burn,thrust_n=burn.thrust_n*(1+magnitude),direction_icrf=tuple(Rotation.from_rotvec(pointing).apply(burn.direction_icrf)))
            rollout=plant.propagate(actual,t,t+h,mass,executed)
            records.append(dict(time_s=float(t),dv=dv.tolist(),burn_s=0. if burn is None else burn.stop_s-burn.start_s,status=status,mode=mode))
            if not rollout.completed:collision=True;break
            for stamp in np.arange(t+300,t+h+1,300):
                dense.append(dict(time_s=float(stamp),error_m=float(np.linalg.norm(rollout.at(stamp)[:3]-plan.at(stamp)[:3]))))
            actual,mass=rollout.at(t+h)[:6],rollout.at(t+h)[6];history=rollout
        steady=np.array([x['error_m'] for x in dense if x['time_s']>=DAY/2]);errors=np.array([x['error_m'] for x in dense])
        fired=[r for r in records if r['burn_s']>0]
        return dict(scenario=asdict(scenario),profile=profile,arithmetic='float64' if library is None else '32',
            classification='Synthetic controlled vehicle; one reconstructed ARTEMIS spacecraft initial state; no future spacecraft samples',
            initial_state=initial.tolist(),model=inputs,nominal_min_clearance_m=plan.min_clearance_m,
            filter_settings=asdict(spec['filter']),guard=None if spec['guard'] is None else asdict(spec['guard']),
            metrics=dict(rms_m=float(np.sqrt(np.mean(steady**2))) if len(steady) else None,peak_m=float(max(steady)) if len(steady) else None,
                full_rms_m=float(np.sqrt(np.mean(errors**2))),full_peak_m=float(max(errors)),dv_m_s=float(dv_total),burns=burns,
                direction_changes=sum(float(np.dot(a['dv'],b['dv']))<0 for a,b in zip(fired[:-1],fired[1:])),
                collision=collision,completed=bool(not collision and dense[-1]['time_s']>=scenario.days*DAY),
                numerical_faults=sum(statuses.get(str(k),0) for k in (1,4,16,32)),
                supervisor_halts=sum(statuses.get(str(k),0) for k in (64,128)),
                fixes_rejected=nav.rejected,fixes_accepted=nav.accepted,online_cpu_s=online,mode_counts=modes,status_counts=statuses,max_cache_sizes=max_caches),
            total_cpu_s=time.process_time()-start,samples=samples,dense_samples=dense,commands=records)
