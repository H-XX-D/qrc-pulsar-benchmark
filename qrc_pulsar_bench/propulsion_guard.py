"""Independent propulsion interlock driven by timestamped sensor products.

No fault labels, truth state or plant parameters enter PropulsionGuard. The
separate plant adapter below simulates an independent shutoff and backup bank.
"""
from dataclasses import dataclass,replace
import numpy as np


@dataclass(frozen=True)
class PropulsionSample:
    time_s: float
    thrust_n: float
    flow_kg_s: float
    isolation_closed: bool


class PropulsionGuard:
    def __init__(self,*,backup_available=True):
        self.backup_available=backup_available
        self.last_s=-np.inf;self.suspect_count=0;self.quiet_count=0
        self.isolation_requested=False;self.closed_confirmed=False
        self.use_backup=False;self.invalid_samples=0
        self.uncommanded_propellant_kg=0.;self.events=[]

    @property
    def thrust_allowed(self):
        return not self.isolation_requested or self.use_backup

    def observe(self,sample,*,available_s,expected_thrust_n):
        values=[sample.time_s,sample.thrust_n,sample.flow_kg_s,available_s,expected_thrust_n]
        if (not np.all(np.isfinite(values)) or sample.time_s<=self.last_s
                or not 0<=available_s-sample.time_s<=120
                or min(sample.thrust_n,sample.flow_kg_s,expected_thrust_n)<0
                or type(sample.isolation_closed) is not bool):
            self.invalid_samples+=1;return False
        gap=sample.time_s-self.last_s
        if gap>120:self.suspect_count=0;self.quiet_count=0
        self.last_s=float(sample.time_s)
        unexpected=(expected_thrust_n<.005 and sample.thrust_n>.02 and sample.flow_kg_s>1e-5)
        self.suspect_count=self.suspect_count+1 if unexpected else 0
        if unexpected and np.isfinite(gap):
            self.uncommanded_propellant_kg+=sample.flow_kg_s*min(gap,120.)
        if self.suspect_count>=2 and not self.isolation_requested:
            self.isolation_requested=True
            self.events.append(dict(time_s=self.last_s,event='isolate_primary',reason='uncommanded_force_and_flow'))
        if self.isolation_requested:
            self.closed_confirmed=sample.isolation_closed
            quiet=sample.isolation_closed and sample.thrust_n<.01 and sample.flow_kg_s<5e-6
            self.quiet_count=self.quiet_count+1 if quiet else 0
            if self.quiet_count>=2 and self.backup_available and not self.use_backup:
                self.use_backup=True
                self.events.append(dict(time_s=self.last_s,event='backup_enabled',reason='closed_and_quiet_confirmed'))
            if not sample.isolation_closed:self.use_backup=False
        self.events=self.events[-128:]
        return True


class GuardedPlant:
    """Synthetic pressure/thrust telemetry and independently actuated shutoff.

    Sensors sample at60s; two independent scalar indications must agree twice.
    A requested shutoff changes the plant only when the modeled hardware works.
    Integrate exactly at the resulting cutoff, without sixty-second ODE resets.
    """
    def __init__(self,guard,seed,*,isolation_available=True):
        self.guard=guard;self.rng=np.random.default_rng(seed)
        self.isolation_available=isolation_available;self.closed=False
        self.last_cutoff_s=None

    def propagate(self,model,state,start,stop,mass,executed,requested,leak_force,leak_flow):
        from .compound_benchmark import propagate_plant,PlantRollout
        was_closed=self.closed;backup=self.guard.use_backup;cutoff=None
        for t in np.arange(start+60.,stop+1e-7,60.):
            # Left limit: measurements represent force immediately BEFORE a
            # sampled stop. This keeps a one-step leak from disappearing.
            on=executed is not None and t<=executed.stop_s
            commanded=executed.thrust_n if on and (backup or not self.closed) else 0.
            actual_force=np.zeros(3) if self.closed else np.asarray(leak_force).copy()
            if commanded:actual_force+=commanded*np.asarray(executed.direction_icrf)
            flow=(0. if self.closed else leak_flow)+(executed.mass_flow_kg_s if commanded else 0.)
            expected=requested.thrust_n if requested is not None and t<=requested.stop_s else 0.
            if self.closed and not backup:expected=0.
            sample=PropulsionSample(float(t),max(0.,float(np.linalg.norm(actual_force))+self.rng.normal()*0.001),
                max(0.,float(flow)+self.rng.normal()*1e-7),bool(self.closed))
            self.guard.observe(sample,available_s=float(t),expected_thrust_n=expected)
            if self.guard.isolation_requested and self.isolation_available and not self.closed:
                self.closed=True;cutoff=float(t);self.last_cutoff_s=cutoff
        parts=[];net=0.
        cuts=sorted({start,stop,*([cutoff] if cutoff is not None and start<cutoff<stop else [])})
        for left,right in zip(cuts[:-1],cuts[1:]):
            closed=was_closed or (cutoff is not None and left>=cutoff)
            burn=executed if backup or not closed else None
            if burn is not None:
                if burn.stop_s<=left:burn=None
                else:burn=replace(burn,start_s=left,stop_s=min(right,burn.stop_s),mass_start_kg=mass)
            rollout=propagate_plant(model,state,left,right,mass,burn,
                np.zeros(3) if closed else leak_force,0. if closed else leak_flow)
            parts.extend(rollout.parts);net+=rollout.propulsive_dv
            final=rollout.at(rollout.stop_s);state=final[:6];mass=float(final[6])
            if not rollout.completed:break
        return PlantRollout(parts,net),cutoff
