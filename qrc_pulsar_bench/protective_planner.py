"""Bounded finite-burn candidate screening outside routine tracking authority.

The13-point uncertainty ensemble is a diagnostic screen, not a certified
reachable set. The original mission reference is never moved to the estimate.
"""
from dataclasses import dataclass
import numpy as np
from .adaptive_control import covariance_factor


@dataclass(frozen=True)
class ProtectiveSettings:
    horizon_s: float = 21600.
    maximum_dv_m_s: float = .15
    position_uncertainty_m: float = 200.
    velocity_uncertainty_m_s: float = .02
    sigma_multiplier: float = 6.
    moon_floor_m: float = 50000.
    earth_floor_m: float = 200000.


class ProtectivePlanner:
    def __init__(self,model,plan,actuator,settings=ProtectiveSettings()):
        self.model,self.plan,self.actuator,self.settings=model,plan,actuator,settings
        values=list(vars(settings).values())
        if not np.all(np.isfinite(values)) or min(values)<=0:raise ValueError('Positive finite protection settings required')
        self.last_report={}

    def command(self,t,estimate,mass,remaining_dv,proposal=None,*,allow_recovery=False):
        self.last_report=dict(feasible=False,reason='invalid_navigation',candidates=0,rollouts=0)
        try:
            state=np.asarray(estimate['state'],float);p=np.asarray(estimate['covariance'],float)
            s=self.settings
            if (estimate.get('navigation_trusted') is not True or state.shape!=(9,) or p.shape!=(9,9)
                    or not np.all(np.isfinite(state)) or abs(estimate['time_s']-t)>1e-6
                    or not 0<=t-estimate['last_observation_s']<=3600
                    or not np.all(np.isfinite([t,mass,remaining_dv])) or remaining_dv<0):return np.zeros(3)
            root=covariance_factor(p[:6,:6])
            if (3*np.sqrt(np.linalg.eigvalsh(p[:3,:3])[-1])>s.position_uncertainty_m
                    or 3*np.sqrt(np.linalg.eigvalsh(p[3:6,3:6])[-1])>s.velocity_uncertainty_m_s):return np.zeros(3)
            end=min(t+s.horizon_s,self.plan.stop_s)
            if end-t<1:return np.zeros(3)
            cap=min(s.maximum_dv_m_s,remaining_dv,self.actuator.capacity(mass,interval_s=min(1800.,end-t)))
            candidates=[np.zeros(3)]
            if proposal is not None:
                proposal=np.asarray(proposal,float)
                if proposal.shape==(3,) and np.all(np.isfinite(proposal)) and 0<np.linalg.norm(proposal)<=cap+1e-12:
                    candidates.append(proposal)
            for axis in np.eye(3):
                if cap>=self.actuator.minimum_dv_m_s:candidates.extend([axis*cap,-axis*cap])
            points=[state[:6]]+[state[:6]+sign*s.sigma_multiplier*root[:,i] for i in range(6) for sign in (-1,1)]
            options=[];total_rollouts=0
            for dv in candidates:
                burn=self.actuator.command(dv,t,mass,interval_s=min(1800.,end-t))
                safe=True;worst=0.;minimum=np.inf
                for point in points:
                    rollout=self.model.propagate(point,t,end,mass,burn,floors=(s.moon_floor_m,s.earth_floor_m))
                    total_rollouts+=1;minimum=min(minimum,rollout.min_clearance_m)
                    if not rollout.completed or rollout.min_clearance_m<=0:safe=False;break
                    terminal=rollout.at(end)
                    if terminal[6]<self.actuator.dry_mass_kg+self.actuator.reserve_kg:safe=False;break
                    delta=terminal[:6]-self.plan.at(end)[:6]
                    worst=max(worst,float(np.linalg.norm(delta[:3])+1800*np.linalg.norm(delta[3:])))
                if safe:
                    options.append((float(np.linalg.norm(dv)),worst,dv,minimum))
                    # Protection prioritizes preservation of clearance/reserve.
                    # Do not spend emergency authority chasing a distant
                    # reference when the screened coast already meets them.
                    if not allow_recovery and not np.linalg.norm(dv):break
            self.last_report.update(candidates=len(candidates),rollouts=total_rollouts,horizon_end_s=float(end))
            if not options:
                self.last_report['reason']='no_screened_candidate';return np.zeros(3)
            if allow_recovery and proposal is not None:
                coast_opt=next((opt for opt in options if opt[0]==0.),None)
                prop_norm=float(np.linalg.norm(proposal))
                prop_opt=next((opt for opt in options if abs(opt[0]-prop_norm)<1e-9 and np.allclose(opt[2],proposal,atol=1e-8)),None)
                if prop_opt is not None and (coast_opt is None or prop_opt[1]<coast_opt[1]-1e-6):
                    effort,cost,dv,minimum=prop_opt
                else:
                    effort,cost,dv,minimum=min(options,key=lambda x:(x[0],x[1]))
            else:
                effort,cost,dv,minimum=min(options,key=lambda x:(x[0],x[1]))
            self.last_report.update(feasible=True,reason='screened_burn' if np.linalg.norm(dv) else 'screened_coast',
                score=float(cost),minimum_sampled_clearance_margin_m=float(minimum),ensemble_points=13)
            return dv.copy()
        except (ValueError,KeyError,TypeError,RuntimeError,np.linalg.LinAlgError) as exc:
            self.last_report.update(reason='screen_failed',detail=str(exc));return np.zeros(3)
