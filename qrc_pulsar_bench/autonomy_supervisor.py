"""Orthogonal health monitoring and bounded ground-supervised MPC recovery."""
from collections import deque
import numpy as np
from .routine_supervisor import RoutineMPC
from .mpc_strategies import StrategyMPC,strategies
from .protective_planner import ProtectivePlanner


class HealthMonitor:
    def __init__(self):
        self.states={};self.events=deque(maxlen=256);self.updates=0

    def observe(self,t,estimate,guard,ground_up,authority,actuator):
        states=dict(navigation='trusted' if estimate.get('navigation_trusted') is True and
            0<=t-estimate['last_observation_s']<=3600 else 'untrusted',
            propulsion=('backup' if guard.use_backup else 'isolation_pending') if guard.isolation_requested else 'primary',
            ground='connected' if ground_up else 'blackout',
            mission='enabled' if authority.state['mission_enabled'] else 'held',
            reserve='available' if estimate['mass']>actuator.dry_mass_kg+actuator.reserve_kg else 'depleted')
        for key,value in states.items():
            if self.states.get(key)!=value:self.events.append(dict(time_s=float(t),subsystem=key,state=value))
        self.states=states;self.updates+=1
        return dict(states)


class AutonomousMPC:
    def __init__(self,model,plan,guard,authority,*,library=None):
        self.routine=RoutineMPC(model,plan,library=library,bits=32)
        self.recovery=StrategyMPC(model,plan,strategies()['sparse_05'],library=library,bits=32)
        self.settings,self.actuator=self.routine.settings,self.routine.actuator
        self.planner=ProtectivePlanner(model,plan,self.actuator)
        self.guard,self.authority=guard,authority
        self.health=HealthMonitor();self.ground_up=True
        self.issued=deque();self.last_epoch=-np.inf;self.last_status=0;self.mode='initializing'
        self.protective_reports=deque(maxlen=256);self.inside_since=None

    def coast(self,status,mode):
        self.last_status=status;self.mode=mode;return np.zeros(3)

    def command(self,t,estimate,mass,*,valid=True):
        # Every monitor updates even when another subsystem already inhibits thrust.
        try:self.health.observe(t,estimate,self.guard,self.ground_up,self.authority,self.actuator)
        except (KeyError,TypeError,ValueError):return self.coast(1,'invalid_health_input')
        if not np.isfinite(t) or t<self.last_epoch+self.settings.interval_s-1e-6:return self.coast(1,'decision_epoch_invalid')
        self.last_epoch=float(t)
        try:self.routine.validate_input(t,estimate,mass,record_epoch=False)
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):return self.coast(1,'invalid_navigation')
        if not valid or self.health.states['navigation']!='trusted':
            self.inside_since=None;return self.coast(2,'navigation_recovery')
        if not self.guard.thrust_allowed:
            self.inside_since=None;return self.coast(1024,'propulsion_isolation')
        if self.health.states['reserve']!='available':return self.coast(1024,'reserve_depleted')
        while self.issued and self.issued[0][0]<=t-86400:self.issued.popleft()
        remaining=max(0.,self.authority.state['daily_limit_m_s']-sum(d for _,d in self.issued))
        position,velocity=self.routine.tracking_error(t,estimate)
        inside=position<2500 and velocity<.25
        self.inside_since=(t if self.inside_since is None else self.inside_since) if inside else None
        if self.routine.recovery_required and self.inside_since is not None and t-self.inside_since>=7200:
            self.routine.rearm(t,estimate,mass)
        mission=self.authority.state['mission_enabled']
        previous_issued=list(self.routine.issued);previous_burn=self.routine.last_burn
        if mission:
            dv=self.routine.command(t,estimate,mass)
            status=self.routine.last_status;mode=self.routine.mode
        else:dv=np.zeros(3);status=2048;mode='ground_mission_hold'
        if status in (32,64) or not mission:
            proposal=None
            if mission:
                proposal=self.recovery.command(t,estimate,mass)
                if self.recovery.last_status:proposal=None
            # Under a ground hold, preserve emergency protection but suppress
            # mission-return burns whenever the screened coast is feasible.
            if not mission:
                decision=self.planner.command(t,estimate,mass,0.)
                report=dict(self.planner.last_report)
                if not report['feasible']:decision=self.planner.command(t,estimate,mass,remaining)
            else:decision=self.planner.command(t,estimate,mass,remaining,proposal,allow_recovery=True)
            report=dict(self.planner.last_report,time_s=float(t))
            self.protective_reports.append(report)
            if not report['feasible']:return self.coast(4096,'no_screened_protective_plan')
            dv=decision;status=0;mode='protective_'+report['reason']
        if np.linalg.norm(dv)>remaining+1e-12:
            self.routine.issued=deque(previous_issued);self.routine.last_burn=previous_burn
            return self.coast(128,'shared_budget_wait')
        if status:return self.coast(status,mode)
        if np.linalg.norm(dv) and self.actuator.command(dv,t,mass,interval_s=self.settings.interval_s) is not None:
            self.issued.append((float(t),float(np.linalg.norm(dv))))
        self.last_status=0;self.mode=mode
        return dv
