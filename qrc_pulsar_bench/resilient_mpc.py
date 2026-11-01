"""Bounded degraded-navigation operation around the sparse MPC controller.

Covariance gates are engineering assumptions, not calibrated probability or
collision guarantees. A caller's invalid-health flag always inhibits output.
"""
from dataclasses import dataclass, replace
import numpy as np
from scipy.linalg import expm
from .mpc_strategies import StrategyMPC, strategies
from .adaptive_control import FilterSettings


@dataclass(frozen=True)
class NavigationGuard:
    fresh_age_s: float = 3600.
    maximum_age_s: float = 8*3600.
    position_sigma_limit_m: float = 200.
    velocity_sigma_limit_m_s: float = .03
    degraded_dv_limit_m_s: float = .03
    clearance_sigma_factor: float = 3.
    bias_correlation_s: float = 0.

    def __post_init__(self):
        if not np.all(np.isfinite(list(vars(self).values()))) or min(vars(self).values())<0:
            raise ValueError('Finite nonnegative guard settings required')
        if self.maximum_age_s<self.fresh_age_s or min(self.maximum_age_s,self.position_sigma_limit_m,self.velocity_sigma_limit_m_s,self.degraded_dv_limit_m_s)<=0:
            raise ValueError('Invalid navigation guard bounds')


def profiles():
    """Declared development options. No held-out state is used to choose them."""
    return {
        'legacy': dict(filter=FilterSettings(), guard=None, bias=False),
        'decay_only': dict(filter=FilterSettings(bias_correlation_s=7200.), guard=None, bias=False),
        'guarded': dict(filter=FilterSettings(), guard=NavigationGuard(), bias=True),
        'decay_guarded': dict(filter=FilterSettings(bias_correlation_s=7200.),
                              guard=NavigationGuard(bias_correlation_s=7200.), bias=True),
    }


class GuardedMPC(StrategyMPC):
    def __init__(self, model, plan, guard=NavigationGuard(), **kwargs):
        super().__init__(model,plan,replace(strategies()['sparse_05'],use_bias=True),**kwargs)
        self.guard=guard;self.mode='inhibited'

    def transition(self,t,h):
        key=(float(t),float(h))
        if key not in self.affine_cache:
            a=np.zeros((9,9));a[:6,:6]=self._linear(t,h);a[3:6,6:]=np.eye(3)
            if self.guard.bias_correlation_s:a[6:,6:]=-np.eye(3)/self.guard.bias_correlation_s
            self.affine_cache[key]=expm(a*h)
            if len(self.affine_cache)>2*self.settings.horizon_steps+4:
                del self.affine_cache[next(iter(self.affine_cache))]
        return self.affine_cache[key]

    def command(self,t,estimate,mass,*,valid=True):
        self.mode='inhibited'
        if not valid:return self._coast(t,'invalid_navigation_health',2)
        try:
            state=self.validate_input(t,estimate,mass,record_epoch=False)
            if state.shape!=(9,) or 'time_s' not in estimate or 'last_observation_s' not in estimate:
                raise ValueError('Nine-state estimate and both timestamps required')
            covariance=np.asarray(estimate['covariance'],dtype=float)
            age=t-estimate['last_observation_s']
            sigma=np.sqrt(max(0.,np.linalg.eigvalsh(covariance[:3,:3])[-1]))
            velocity=np.sqrt(max(0.,np.linalg.eigvalsh(covariance[3:6,3:6])[-1]))
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):return self._coast(None,'invalid_input',1)
        limits=self.guard
        if age>limits.maximum_age_s or sigma>limits.position_sigma_limit_m or velocity>limits.velocity_sigma_limit_m_s:
            return self._coast(t,'navigation_guard',2,age_s=age,position_sigma_m=sigma,velocity_sigma_m_s=velocity)
        nominal=self.settings
        self.mode='degraded' if age>limits.fresh_age_s else 'fresh'
        cap=min(nominal.maximum_dv_m_s,limits.degraded_dv_limit_m_s) if self.mode=='degraded' else nominal.maximum_dv_m_s
        # Inflate geometric body floors by the current position uncertainty.
        self.settings=replace(nominal,maximum_dv_m_s=cap,
                              moon_floor_m=nominal.moon_floor_m+limits.clearance_sigma_factor*sigma,
                              earth_floor_m=nominal.earth_floor_m+limits.clearance_sigma_factor*sigma)
        try:
            command=super().command(t,estimate,mass,valid=True)
            self.diagnostics[-1].update(mode=self.mode,age_s=age,position_sigma_m=sigma,velocity_sigma_m_s=velocity,dv_limit_m_s=cap)
            if self.last_status:self.mode='inhibited'
            return command
        finally:self.settings=nominal
