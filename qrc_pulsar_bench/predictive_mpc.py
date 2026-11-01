"""Receding-horizon control with bounded QP work and no PID fallback.

The optimizer runs on tracking-error dynamics. Restricted Jacobi translation
does not change those errors. The host builds time-varying force Jacobians;
only the QP solver is the portable integer HLS candidate.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm
from .adaptive_control import force_jacobian,validate_covariance
from .jacobi_control import Actuator
from .fpga_mpc import FixedQP,scaled_qp,float_solve,MAX_N


@dataclass(frozen=True)
class MPCSettings:
    interval_s: float = 1800.
    horizon_steps: int = 12
    position_scale_m: float = 100.
    velocity_scale_m_s: float = .01
    dv_scale_m_s: float = .05
    maximum_dv_m_s: float = .15
    residual_tolerance: float = 2e-5
    moon_floor_m: float = 50000.
    earth_floor_m: float = 200000.

    def __post_init__(self):
        if not isinstance(self.horizon_steps,int) or not 1<=self.horizon_steps<=MAX_N//3:raise ValueError('Horizon exceeds fixed workspace')
        if not np.all(np.isfinite(list(vars(self).values()))) or min(vars(self).values())<=0:raise ValueError('Positive finite settings required')


class BoundedLog(list):
    """Recent onboard diagnostics; campaigns collect their own full history."""
    def append(self,item):
        super().append(item)
        if len(self)>256:del self[:-256]


def finite_burn_transition(a,interval,duration):
    """Exact constant-linear-system transition for a rectangular early burn.

    Control is total delta-v, not acceleration. Duration=0 is an impulse.
    Actual changing gravity is approximated locally; actual burns are checked
    with the nonlinear propagator before the first command is accepted.
    """
    if not 0<=duration<=interval:raise ValueError('Burn outside interval')
    phi=expm(a*interval)
    if duration<=1e-9:return phi,phi[:,3:]
    augmented=np.zeros((9,9));augmented[:6,:6]=a;augmented[3:6,6:]=np.eye(3)
    response=expm(augmented*duration)[:6,6:]/duration
    return phi,expm(a*(interval-duration))@response


class PredictiveMPC:
    def __init__(self,model,plan,settings=MPCSettings(),*,library=None,bits=32,actuator=Actuator(minimum_dv_m_s=.0001)):
        self.model,self.plan,self.settings,self.actuator=model,plan,settings,actuator
        self.solver=None if library is None else FixedQP(library,bits,bits-8)
        self.warm=np.zeros((settings.horizon_steps,3));self.cache={};self.last_status=0
        self.diagnostics=BoundedLog();self.last_schedule=np.empty((0,3));self.last_epoch=-np.inf

    def validate_input(self,t,estimate,mass,*,record_epoch=True):
        if not np.all(np.isfinite([t,mass])) or t<0 or t>=self.plan.stop_s or mass<=0:
            raise ValueError('Invalid decision epoch or mass')
        if t<=self.last_epoch:raise ValueError('Duplicate or reversed decision epoch')
        state=np.asarray(estimate['state'],dtype=float)
        if state.shape not in ((6,),(9,)) or not np.all(np.isfinite(state)):
            raise ValueError('Invalid navigation state')
        if 'time_s' in estimate and (not np.isfinite(estimate['time_s']) or abs(estimate['time_s']-t)>1e-6):
            raise ValueError('Estimate is not propagated to the decision epoch')
        if 'last_observation_s' in estimate and (not np.isfinite(estimate['last_observation_s']) or estimate['last_observation_s']>t):
            raise ValueError('Invalid observation epoch')
        if 'covariance' in estimate:
            if np.shape(estimate['covariance'])!=(len(state),len(state)):raise ValueError('Covariance shape')
            validate_covariance(estimate['covariance'])
        if record_epoch:self.last_epoch=float(t)
        return state

    def inhibit(self,status,reason,t=None):
        self.last_status=status;self.warm[:]=0;self.last_schedule=np.empty((0,3))
        self.diagnostics.append(dict(time_s=t,status=status,reason=reason))
        return np.zeros(3)

    def _linear(self,t,h):
        key=(float(t),float(h))
        if key not in self.cache:
            state=self.model.to_moon(self.plan.at(t+h/2)[:6],t+h/2)
            self.cache[key]=force_jacobian(self.model.acceleration,state,t+h/2)
            if len(self.cache)>2*self.settings.horizon_steps+4:del self.cache[next(iter(self.cache))]
        return self.cache[key]

    def build_problem(self,t,state,mass,durations):
        s=self.settings;count=len(durations);n=3*count
        total=self.actuator.exhaust_velocity*np.log(mass/min(mass,self.actuator.dry_mass_kg+self.actuator.reserve_kg))
        cap=min(s.maximum_dv_m_s,self.actuator.capacity(mass,interval_s=s.interval_s),total/count)/np.sqrt(3)
        if cap<=0:raise ValueError('No routine thrust authority')
        error=state-self.plan.at(t)[:6];response=np.zeros((6,n))
        scale=np.r_[[1/s.position_scale_m]*3,[1/s.velocity_scale_m_s]*3]
        matrices=[];offsets=[]
        for i,duration in enumerate(durations):
            left=t+i*s.interval_s;h=min(s.interval_s,self.plan.stop_s-left)
            phi,b=finite_burn_transition(self._linear(left,h),h,duration)
            response=phi@response;response[:,3*i:3*i+3]+=b*cap
            error=phi@error
            matrices.append(scale[:,None]*response);offsets.append(scale*error)
        matrix=np.vstack([*matrices,np.eye(n)*cap/s.dv_scale_m_s])
        target=-np.r_[np.concatenate(offsets),np.zeros(n)]
        return matrix,target,cap

    def command(self,t,estimate,mass,*,valid=True):
        s=self.settings;self.last_schedule=np.empty((0,3))
        if not valid:return self.inhibit(2,'invalid_navigation',t)
        try:state=self.validate_input(t,estimate,mass)[:6]
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):return self.inhibit(1,'invalid_input')
        remaining=self.plan.stop_s-t
        count=min(s.horizon_steps,int(np.ceil(remaining/s.interval_s)))
        if count<1:self.last_status=1;return np.zeros(3)
        warm=self.warm[:count].copy();durations=np.zeros(count);residual=0.;cap=0.
        try:
            # Exactly two refinements: impulse plan, then finite-burn timing.
            for refinement in range(2):
                matrix,target,cap=self.build_problem(t,state,mass,durations)
                h,g,beta=scaled_qp(matrix,target)
                start=np.clip(warm.ravel()/cap,-1,1)
                if self.solver is None:status,u,residual=float_solve(h,g,start,beta,s.residual_tolerance)
                else:status,u,residual=self.solver.solve(h,g,start,beta,s.residual_tolerance)
                if status:
                    return self.inhibit(status,'solver_failure',t)
                warm=u.reshape(count,3)*cap
                durations=np.array([0. if (b:=self.actuator.command(d,t,mass,interval_s=s.interval_s)) is None else b.stop_s-b.start_s for d in warm])
            proposed=self.actuator.command(warm[0],t,mass,interval_s=min(s.interval_s,remaining))
            # Finite first-burn clearance screen, not a global collision proof.
            rollout=self.model.propagate(state,t,t+min(s.interval_s,remaining),mass,proposed,
                floors=(s.moon_floor_m,s.earth_floor_m))
            if not rollout.completed:
                return self.inhibit(32,'clearance_screen',t)
        except (ValueError,np.linalg.LinAlgError,RuntimeError):
            return self.inhibit(1,'invalid_problem',t)
        self.last_status=0;self.last_schedule=warm.copy()
        self.warm[:]=0
        if count>1:self.warm[:count-1]=warm[1:]
        self.diagnostics.append(dict(time_s=t,status=0,residual=residual,steps=count,
            scheduled_dv_m_s=warm.tolist(),predicted_clearance_m=rollout.min_clearance_m))
        return warm[0]
