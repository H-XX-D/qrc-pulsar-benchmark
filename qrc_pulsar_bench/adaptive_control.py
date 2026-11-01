"""Causal navigation refinement and matched-coordinate control variants.

Synthetic direct-state fixes remain an explicit sensor assumption. The filter
estimates a slowly varying inertial acceleration bias; it does not discover all
missing physics, infer future burns or produce a certified uncertainty tube.
"""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import expm, solve_triangular
from scipy.optimize import lsq_linear
from .jacobi_control import (JacobiFlightModel, JacobiPredictiveController,
                             NavigationEstimate, _array)
from .maneuvers import FiniteBurn


class MoonCartesianFlightModel(JacobiFlightModel):
    """Same forces, integration and body guards in Moon-relative Cartesian axes."""
    def __init__(self, acceleration, inner, **kwargs):
        super().__init__(acceleration, inner, **kwargs)
        self.factor = 0.

    def to_moon(self, state, t):
        return _array(state, (6,), "Moon-relative Cartesian state")

    def from_moon(self, state, t):
        return _array(state, (6,), "Moon-relative Cartesian state")


def force_jacobian(acceleration, state, t):
    x = _array(state, (6,), "linearization state")
    a = np.zeros((6, 6)); a[:3, 3:] = np.eye(3)
    for j in range(6):
        d = np.zeros(6); d[j] = 10. if j < 3 else .001
        p, m = x+d, x-d
        a[3:, j] = (acceleration(p[:3],p[3:],t)-acceleration(m[:3],m[3:],t))/(2*d[j])
    return a


def covariance_step(a, spectral, dt):
    """Exact constant-linear-system covariance step (Van Loan exponential)."""
    n = len(a)
    block = np.zeros((2*n, 2*n))
    block[:n,:n], block[:n,n:], block[n:,n:] = a, spectral, -a.T
    exponential = expm(dt*block)
    phi = exponential[:n,:n]
    q = exponential[:n,n:]@phi.T
    return phi, .5*(q+q.T)


def validate_covariance(p):
    """PSD check in correlation units, including tiny acceleration variances."""
    p=np.asarray(p,dtype=float)
    if p.ndim!=2 or p.shape[0]!=p.shape[1] or not np.all(np.isfinite(p)):
        raise ValueError('Finite square covariance required')
    diagonal=np.diag(p)
    if np.any(diagonal<0):raise ValueError('Negative variance')
    scale=np.sqrt(np.where(diagonal>0,diagonal,1.))
    normalized=p/np.outer(scale,scale)
    if not np.allclose(normalized,normalized.T,rtol=1e-10,atol=1e-12) or np.linalg.eigvalsh(normalized)[0]<-1e-10:
        raise ValueError('Symmetric positive semidefinite covariance required')
    return p


def covariance_factor(p):
    """Factor a validated input covariance in correlation units.

    The eigen fallback handles semidefinite inputs (including zero process
    noise). Only roundoff-sized negative eigenvalues are allowed there. A live
    filter retains its factor and never refactors a rounded full covariance.
    """
    p=validate_covariance(p)
    scale=np.sqrt(np.where(np.diag(p)>0,np.diag(p),1.))
    correlation=p/np.outer(scale,scale)
    try:return scale[:,None]*np.linalg.cholesky(correlation)
    except np.linalg.LinAlgError:
        values,vectors=np.linalg.eigh(correlation)
        tolerance=64*np.finfo(float).eps*len(p)*max(1.,np.max(np.abs(values)))
        if values[0]<-tolerance:raise ValueError('Input covariance cannot be factored reliably')
        return scale[:,None]*(vectors*np.sqrt(np.maximum(values,0.)))


def triangular_factor(columns):
    """QR factor of a sum of covariance outer products, without forming P."""
    if not np.all(np.isfinite(columns)):raise ValueError('Nonfinite covariance factor')
    scale=np.linalg.norm(columns,axis=1)
    scale=np.where(scale>0,scale,1.)
    _,r=np.linalg.qr((columns/scale[:,None]).T,mode='reduced')
    root=scale[:,None]*r.T
    if not np.all(np.isfinite(root)):raise ValueError('Nonfinite covariance factor')
    return root


def square_root_measurement(root,measurement_covariance,innovation):
    """Joint QR conditioning for a fix of the first m state components.

    Returns a nonnegative NIS, mean correction and posterior factor. The joint
    array preserves small measurement noise even when adding R to HPH' would
    round it away. It avoids covariance subtraction and an indefinite S solve.
    """
    m=len(innovation);n=len(root)
    r=covariance_factor(measurement_covariance)
    joint=np.zeros((m+n,m+root.shape[1]))
    joint[:m,:m]=r;joint[:m,m:]=root[:m,:];joint[m:,m:]=root
    factor=triangular_factor(joint)
    whitened=solve_triangular(factor[:m,:m],innovation,lower=True)
    nis=float(whitened@whitened)
    correction=factor[m:,:m]@whitened
    if not np.isfinite(nis) or not np.all(np.isfinite(correction)):
        raise ValueError('Nonfinite navigation update')
    return nis,correction,factor[m:,m:]


def propagate_known_commands(model, state, start, stop, mass, commands=(), *, as_of_s):
    """Reconstruct past command effects with an explicit knowledge-time bound.

    Returns state, mass, and the physical model's finite-burn rollout segments.
    A command issued after `start` may be used only when it is already known at
    `as_of_s`. This differs deliberately from an initial-only blind forecast.
    """
    state = _array(state, (6,), "state")
    if (not np.all(np.isfinite([start,stop,mass,as_of_s])) or stop < start or mass <= 0):
        raise ValueError("Invalid command propagation interval")
    active = sorted((b for b in commands if b.stop_s>start and b.start_s<stop),key=lambda b:b.start_s)
    if any(b.available_at_s > as_of_s for b in active):
        raise ValueError("Future command information is unavailable")
    if any(a.stop_s>b.start_s for a,b in zip(active[:-1],active[1:])):
        raise ValueError("Overlapping command ledger")
    t = start; pieces = []
    def advance(end, burn=None):
        nonlocal state, mass, t
        if end <= t: return
        part = model.propagate(state,t,end,mass,burn)
        if not part.completed: raise RuntimeError("Body collision during navigation propagation")
        y = part.at(end); state, mass, t = y[:6], y[6], end
        pieces.append(part)
    for command in active:
        advance(max(t, command.start_s))
        end = min(stop, command.stop_s)
        part = FiniteBurn(t,end,command.thrust_n,command.direction_icrf,mass,
                          command.mass_flow_kg_s,t,"accepted command in causal state reconstruction")
        advance(end,part)
    advance(stop)
    return state, mass, pieces


@dataclass(frozen=True)
class FilterSettings:
    # Zero preserves the original constant-bias model. Positive values model
    # disturbance decorrelation, including while observations are missing.
    bias_correlation_s: float = 0.
    bias_random_walk_psd: float = 1e-18
    acceleration_noise_psd: float = 1e-10
    initial_bias_sigma_m_s2: float = 1e-7
    maximum_bias_m_s2: float = 2e-6
    nis_limit: float = 35.
    covariance_step_s: float = 1800.
    maneuver_fraction_sigma: float = .05
    maneuver_pointing_sigma_rad: float = .02

    def __post_init__(self):
        values = np.array(list(vars(self).values()))
        if not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Finite nonnegative filter settings required")
        if min(self.maximum_bias_m_s2,self.nis_limit,self.covariance_step_s)<=0:
            raise ValueError("Positive filter limit and step required")


class CausalOrbitFilter:
    """Nine-state EKF: Moon-relative r,v and an inertial acceleration bias.

    Commits only through measurement epochs. A decision forecast is a separate
    object, so a delayed next measurement can be processed without double use
    or time reversal. Covariance uses piecewise midpoint linearizations along
    the predicted mean, not an exact nonlinear distribution.
    """
    def __init__(self, model, state, covariance, mass, *, time_s=0., settings=FilterSettings()):
        if not np.all(np.isfinite([mass,time_s])) or mass<=0:
            raise ValueError('Finite filter epoch and positive mass required')
        self.model = MoonCartesianFlightModel(model.acceleration,model.inner,
            moon_radius_m=model.moon_radius_m,earth_radius_m=model.earth_radius_m,
            max_step_s=model.max_step_s,rtol=model.rtol)
        self.settings = settings
        self.state = np.r_[_array(state,(6,),"filter state"),np.zeros(3)]
        p = _array(covariance,(6,6),"filter covariance")
        full=np.zeros((9,9));full[:6,:6]=p
        full[6:,6:]=np.eye(3)*settings.initial_bias_sigma_m_s2**2
        self._root=covariance_factor(full)
        self.mass, self.time_s = float(mass), float(time_s)
        self.last_observation_s = float(time_s)
        self.last_processed_observation_s = -np.inf
        self.rejected = 0; self.accepted = 0; self.bias_clips = 0
        self._check_covariance(self.covariance)

    @staticmethod
    def _check_covariance(p):
        validate_covariance(p)

    @property
    def covariance(self):
        # A reporting view. The stored factor remains the numerical authority.
        return self._root@self._root.T

    def _predict_factor(self, time_s, commands, as_of_s):
        if not np.all(np.isfinite([time_s,as_of_s])):
            raise ValueError('Finite prediction and knowledge epochs required')
        if time_s < self.time_s: raise ValueError("Out-of-order observation epoch")
        bias = self.state[6:].copy()
        rate=0. if self.settings.bias_correlation_s==0 else 1/self.settings.bias_correlation_s
        model = MoonCartesianFlightModel(lambda r,v,t:self.model.acceleration(r,v,t)+bias*np.exp(-rate*(t-self.time_s)),
            self.model.inner,moon_radius_m=self.model.moon_radius_m,
            earth_radius_m=self.model.earth_radius_m,max_step_s=self.model.max_step_s,rtol=self.model.rtol)
        mean,mass,pieces = propagate_known_commands(model,self.state[:6],self.time_s,time_s,
            self.mass,commands,as_of_s=as_of_s)
        root=self._root.copy()
        spectral = np.zeros((9,9))
        spectral[3:6,3:6]=np.eye(3)*self.settings.acceleration_noise_psd
        spectral[6:,6:]=np.eye(3)*self.settings.bias_random_walk_psd
        for piece in pieces:
            grid=np.linspace(piece.start_s,piece.stop_s,max(2,int(np.ceil((piece.stop_s-piece.start_s)/self.settings.covariance_step_s))+1))
            for left,right in zip(grid[:-1],grid[1:]):
                middle=(left+right)/2;point=piece.at(middle)[:6]
                a=np.zeros((9,9));a[:6,:6]=force_jacobian(self.model.acceleration,point,middle)
                a[3:6,6:]=np.eye(3)
                a[6:,6:]=-rate*np.eye(3)
                phi,q=covariance_step(a,spectral,right-left)
                root=triangular_factor(np.hstack((phi@root,covariance_factor(q))))
            # Approximate finite-burn execution uncertainty by a velocity jump
            # at burn completion, including the half-burn position correlation.
            # Short burns only; this is not a coupled attitude/propulsion model.
            for command in commands:
                if piece.start_s < command.stop_s <= piece.stop_s+1e-7:
                    duration=command.stop_s-command.start_s
                    final_mass=command.mass_start_kg-command.mass_flow_kg_s*duration
                    if command.mass_flow_kg_s:
                        dv=command.thrust_n/command.mass_flow_kg_s*np.log(command.mass_start_kg/final_mass)
                    else:dv=command.thrust_n/command.mass_start_kg*duration
                    u=np.asarray(command.direction_icrf);along=np.outer(u,u)
                    uncertainty=dv*dv*(self.settings.maneuver_fraction_sigma**2*along
                        +self.settings.maneuver_pointing_sigma_rad**2*(np.eye(3)-along))
                    mapping=np.zeros((9,3));mapping[:3,:]=duration/2*np.eye(3);mapping[3:6,:]=np.eye(3)
                    root=triangular_factor(np.hstack((root,mapping@covariance_factor(uncertainty))))
        return np.r_[mean,bias*np.exp(-rate*(time_s-self.time_s))],root,mass

    def _predict(self,time_s,commands,as_of_s):
        state,root,mass=self._predict_factor(time_s,commands,as_of_s)
        covariance=root@root.T
        self._check_covariance(covariance)
        return state,covariance,mass

    def observe(self, state_fix, covariance, measurement_s, available_s, commands=()):
        if not np.all(np.isfinite([measurement_s,available_s])) or available_s<measurement_s:
            raise ValueError("Observation delivered before measurement exists")
        if measurement_s<=self.last_processed_observation_s:
            raise ValueError('Duplicate or replayed observation')
        z=_array(state_fix,(6,),"state fix");r=_array(covariance,(6,6),"measurement covariance")
        self._check_covariance(r)
        if np.min(np.linalg.eigvalsh(r))<=0:raise ValueError("Positive definite measurement covariance required")
        predicted,root,mass=self._predict_factor(measurement_s,commands,available_s)
        innovation=z-predicted[:6]
        nis,correction,posterior=square_root_measurement(root,r,innovation)
        accepted=nis<=self.settings.nis_limit
        clipped=False
        if accepted:
            predicted+=correction
            root=posterior
            norm=np.linalg.norm(predicted[6:])
            if norm>self.settings.maximum_bias_m_s2:
                predicted[6:]*=self.settings.maximum_bias_m_s2/norm
                clipped=True
        # Validate before committing state, factor, timestamps or counters.
        self._check_covariance(root@root.T)
        if not np.all(np.isfinite(predicted)):raise ValueError('Nonfinite navigation state')
        self.state,self._root,self.mass,self.time_s=predicted,root,mass,float(measurement_s)
        self.last_processed_observation_s=float(measurement_s)
        if accepted:
            self.last_observation_s=measurement_s;self.accepted+=1;self.bias_clips+=int(clipped)
        else:self.rejected+=1
        return dict(accepted=bool(accepted),nis=nis,innovation=innovation.copy())

    def forecast(self,time_s,commands=(),*,as_of_s=None):
        as_of_s=time_s if as_of_s is None else as_of_s
        state,p,mass=self._predict(time_s,commands,as_of_s)
        return dict(state=state,covariance=p,mass=mass,time_s=time_s,
                    last_observation_s=self.last_observation_s)


class BiasAwareController(JacobiPredictiveController):
    """Same finite-burn guard, with a causal bias in the prediction and plan."""
    def __init__(self,base_model,reference,reference_end_s,*,method='mpc',**kwargs):
        if method not in ('mpc','pd','target'):raise ValueError("Unknown guidance method")
        self.bias=np.zeros(3);self.method=method
        self.base_acceleration=base_model.acceleration
        cls=MoonCartesianFlightModel if isinstance(base_model,MoonCartesianFlightModel) else JacobiFlightModel
        model=cls(lambda r,v,t:self.base_acceleration(r,v,t)+self.bias,base_model.inner,
            moon_radius_m=base_model.moon_radius_m,earth_radius_m=base_model.earth_radius_m,
            max_step_s=base_model.max_step_s,rtol=base_model.rtol)
        super().__init__(model,reference,reference_end_s,**kwargs)
        self._affine_cache={}

    def set_bias(self,bias):
        self.bias=_array(bias,(3,),"estimated acceleration bias")

    def _damped_dv(self,estimate):
        return super()._damped_dv(estimate)-self.settings.decision_interval_s*self.bias

    def _affine_transition(self,t,h):
        key=(float(t),float(h))
        if key not in self._affine_cache:
            a=np.zeros((9,9));a[:6,:6]=self._linearization(t,h);a[3:6,6:]=np.eye(3)
            e=expm(h*a);self._affine_cache[key]=(e[:6,:6],e[:6,6:])
        return self._affine_cache[key]

    def _plan(self,estimate,mass,end):
        s=self.settings;t=estimate.time_s
        horizon=min(s.planning_horizon_s,self.reference_end_s-t)
        nodes=np.arange(t,t+horizon-1e-6,s.decision_interval_s);n=len(nodes)
        response=np.zeros((6,3*n));error=estimate.state-self.reference(t)
        scale=np.r_[[1/s.position_cost_scale_m]*3,[1/s.velocity_cost_scale_m_s]*3]
        rows=[];offsets=[]
        for i,left in enumerate(nodes):
            response[3:,3*i:3*i+3]+=np.eye(3)
            right=min(left+s.decision_interval_s,t+horizon)
            grid=np.linspace(left,right,max(2,int(np.ceil((right-left)/s.linear_step_s))+1))
            for a,b in zip(grid[:-1],grid[1:]):
                phi,g=self._affine_transition(a,b-a)
                response=phi@response;error=phi@error+g@self.bias
            rows.append(scale[:,None]*response);offsets.append(scale*error)
        matrix=np.vstack([*rows,np.eye(3*n)/s.dv_cost_scale_m_s])
        target=-np.r_[np.concatenate(offsets),np.zeros(3*n)]
        total=self.actuator.exhaust_velocity*np.log(mass/min(mass,self.actuator.dry_mass_kg+self.actuator.reserve_kg))
        cap=min(self.actuator.capacity(mass,interval_s=end-t),total/n)/np.sqrt(3)
        if cap<=0:return nodes,np.zeros((n,3))
        fit=lsq_linear(matrix,target,bounds=(-cap,cap),tol=1e-9,max_iter=100)
        if not fit.success or not np.all(np.isfinite(fit.x)):raise RuntimeError("Affine planner failed")
        return nodes,fit.x.reshape(n,3)

    def _candidate_burns(self,estimate,mass,end):
        if self.method=='pd':
            return np.empty(0),np.empty((0,3)),[(self._damped_dv(estimate),'DAMPED_BURN')]
        if self.method=='target':
            # Conventional single target-point correction. Propagate the
            # sensitivity to one initial impulse, then solve the weighted
            # position/velocity target problem with the same fuel regularizer.
            s=self.settings;t=estimate.time_s
            horizon=min(s.planning_horizon_s,self.reference_end_s-t)
            error=estimate.state-self.reference(t)
            response=np.vstack([np.zeros((3,3)),np.eye(3)])
            grid=np.linspace(t,t+horizon,max(2,int(np.ceil(horizon/s.linear_step_s))+1))
            for a,b in zip(grid[:-1],grid[1:]):
                phi,g=self._affine_transition(a,b-a)
                response=phi@response;error=phi@error+g@self.bias
            scale=np.r_[[1/s.position_cost_scale_m]*3,[1/s.velocity_cost_scale_m_s]*3]
            matrix=np.vstack([scale[:,None]*response,np.eye(3)/s.dv_cost_scale_m_s])
            target=-np.r_[scale*error,np.zeros(3)]
            cap=self.actuator.capacity(mass,interval_s=end-t)/np.sqrt(3)
            dv=np.zeros(3) if cap<=0 else lsq_linear(matrix,target,bounds=(-cap,cap),tol=1e-9).x
            return np.array([t]),dv[None,:],[(dv,'TARGET_POINT_BURN')]
        return super()._candidate_burns(estimate,mass,end)
