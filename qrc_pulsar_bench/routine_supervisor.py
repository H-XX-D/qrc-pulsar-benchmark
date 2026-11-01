"""Limit routine local MPC authority; no automatic emergency recovery is claimed."""
from dataclasses import dataclass
from collections import deque
import numpy as np
from .mpc_strategies import StrategyMPC,strategies


@dataclass(frozen=True)
class RoutineLimits:
    position_error_m: float = 5000.
    velocity_error_m_s: float = .5
    issued_dv_per_day_m_s: float = 1.
    def __post_init__(self):
        if not np.all(np.isfinite(list(vars(self).values()))) or min(vars(self).values())<=0:
            raise ValueError('Positive finite routine authority limits required')


class RoutineMPC(StrategyMPC):
    """Sparse MPC with an estimated-error envelope and a rolling issued-DV budget.

    Status64 requires an explicit rearm after a new recovery plan/valid estimate.
    Status128 waits for rolling budget headroom. Neither claims a safe coast.
    """
    def __init__(self,model,plan,limits=RoutineLimits(),**kwargs):
        super().__init__(model,plan,strategies()['sparse_05'],**kwargs)
        self.limits=limits;self.issued=deque();self.recovery_required=False;self.mode='inhibited'

    def tracking_error(self,t,estimate):
        error=np.asarray(estimate['state'])[:6]-self.plan.at(t)[:6]
        return float(np.linalg.norm(error[:3])),float(np.linalg.norm(error[3:]))

    def command(self,t,estimate,mass,*,valid=True):
        if self.recovery_required:
            self.mode='recovery_required';return self._coast(t,'recovery_required',64)
        if not valid:
            self.mode='inhibited';return self._coast(t,'invalid_navigation',2)
        try:
            self.validate_input(t,estimate,mass,record_epoch=False)
            if np.shape(estimate.get('covariance'))!=(9,9) or 'time_s' not in estimate or 'last_observation_s' not in estimate:
                raise ValueError('Timestamped nine-state covariance required')
            if t<self.last_epoch+self.settings.interval_s-1e-6:
                raise ValueError('Decision cadence exceeded')
            position,velocity=self.tracking_error(t,estimate)
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):
            self.mode='inhibited';return self._coast(None,'invalid_input',1)
        if t-estimate['last_observation_s']>3600.:
            self.mode='inhibited';return self._coast(t,'stale_navigation',2)
        if position>self.limits.position_error_m or velocity>self.limits.velocity_error_m_s:
            self.recovery_required=True;self.mode='recovery_required'
            return self._coast(t,'outside_routine_tracking_envelope',64,position_error_m=position,velocity_error_m_s=velocity)
        while self.issued and self.issued[0][0]<=t-86400.:self.issued.popleft()
        remaining=max(0.,self.limits.issued_dv_per_day_m_s-sum(dv for _,dv in self.issued))
        # Reject the whole planned command when it cannot fit the budget;
        # post-optimization vector clipping would invalidate its rollout screen.
        self.mode='fresh'
        previous_burn=self.last_burn
        command=super().command(t,estimate,mass,valid=True)
        if self.last_status:self.mode='inhibited';return command
        norm=float(np.linalg.norm(command))
        if norm>remaining+1e-12:
            self.last_burn=previous_burn
            self.mode='budget_wait';return self._coast(t,'routine_delta_v_budget',128,remaining_dv_m_s=remaining)
        if norm and self.actuator.command(command,t,mass,interval_s=self.settings.interval_s) is not None:
            self.issued.append((t,norm))
        return command

    def rearm(self,t,estimate,mass):
        """Explicit caller action; never resets spent budget or decision history."""
        self.validate_input(t,estimate,mass,record_epoch=False)
        if np.shape(estimate.get('covariance'))!=(9,9) or 'time_s' not in estimate or 'last_observation_s' not in estimate or t-estimate['last_observation_s']>3600:
            raise ValueError('Fresh timestamped navigation required to rearm')
        position,velocity=self.tracking_error(t,estimate)
        if position>self.limits.position_error_m or velocity>self.limits.velocity_error_m_s:
            raise ValueError('State remains outside the routine envelope')
        self.recovery_required=False;self.warm[:]=0;self.previous[:]=0
