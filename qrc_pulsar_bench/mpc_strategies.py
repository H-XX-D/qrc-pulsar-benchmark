"""Predictive control tradeoffs: smoothing, sparse effort and coast scheduling.

All policies use causal estimates and a model-generated reference. The existing
bounded integer QP core is reused; host-side scheduling/IRLS are not RTL results.
"""
from dataclasses import dataclass,field,replace
import numpy as np
from scipy.linalg import expm
from .predictive_mpc import MPCSettings,PredictiveMPC
from .fpga_mpc import scaled_qp,float_solve


@dataclass(frozen=True)
class Strategy:
    mpc: MPCSettings = field(default_factory=lambda:MPCSettings(horizon_steps=4))
    slew_scale_m_s: float = 0.
    minimum_spacing_s: float = 0.
    trigger_m: float = 0.
    uncertainty_factor: float = 0.
    sparse_weight: float = 0.
    sparse_epsilon: float = .01
    minimum_command_m_s: float = .0001
    use_bias: bool = False
    dwell_override_m: float = 500.

    def __post_init__(self):
        numbers=[v for k,v in vars(self).items() if k not in ('mpc','use_bias')]
        if not np.all(np.isfinite(numbers)) or min(numbers)<0 or self.sparse_epsilon<=0 or self.dwell_override_m<=0:
            raise ValueError('Invalid strategy settings')
        if self.minimum_spacing_s and not np.isclose(self.minimum_spacing_s/self.mpc.interval_s,round(self.minimum_spacing_s/self.mpc.interval_s)):
            raise ValueError('Burn spacing must align with decision epochs')


def strategies():
    base=Strategy()
    return {
        'baseline':base,
        'smooth_003':replace(base,slew_scale_m_s=.03),
        'smooth_010':replace(base,slew_scale_m_s=.10),
        'spaced_1h':replace(base,minimum_spacing_s=3600.),
        'spaced_2h':replace(base,minimum_spacing_s=7200.,mpc=replace(base.mpc,horizon_steps=8)),
        'event_30m':replace(base,trigger_m=30.),
        'event_60m':replace(base,trigger_m=60.),
        'event_100m':replace(base,trigger_m=100.),
        'sparse_01':replace(base,sparse_weight=.1,minimum_command_m_s=.001),
        'sparse_05':replace(base,sparse_weight=.5,minimum_command_m_s=.001),
        'event60_spaced1h':replace(base,trigger_m=60.,minimum_spacing_s=3600.),
        'bias_predictive':replace(base,use_bias=True),
        'bias_event60':replace(base,use_bias=True,trigger_m=60.),
        'sparse_event60':replace(base,sparse_weight=.1,trigger_m=60.,minimum_command_m_s=.001),
        'uncertain_event60':replace(base,trigger_m=60.,uncertainty_factor=2.),
    }


class StrategyMPC(PredictiveMPC):
    def __init__(self,model,plan,strategy,**kwargs):
        super().__init__(model,plan,strategy.mpc,**kwargs)
        self.strategy=strategy;self.bias=np.zeros(3);self.previous=np.zeros(3)
        self.previous_epoch=-np.inf;self.last_burn=-np.inf;self.affine_cache={}

    def transition(self,t,h):
        key=(float(t),float(h))
        if key not in self.affine_cache:
            a=np.zeros((9,9));a[:6,:6]=self._linear(t,h);a[3:6,6:]=np.eye(3)
            self.affine_cache[key]=expm(a*h)
            if len(self.affine_cache)>2*self.settings.horizon_steps+4:
                del self.affine_cache[next(iter(self.affine_cache))]
        return self.affine_cache[key]

    def coast_peak(self,t,estimate,count):
        error=np.r_[estimate['state'][:6]-self.plan.at(t)[:6],self.bias]
        covariance=np.asarray(estimate['covariance']).copy()
        peak=0.
        for i in range(count):
            left=t+i*self.settings.interval_s;h=min(self.settings.interval_s,self.plan.stop_s-left)
            phi=self.transition(left,h);error=phi@error;covariance=phi@covariance@phi.T
            # Diagnostic confidence allowance, not a nonlinear reachable tube.
            sigma=np.sqrt(max(0.,np.max(np.linalg.eigvalsh(covariance[:3,:3]))))
            peak=max(peak,np.linalg.norm(error[:3])+self.strategy.uncertainty_factor*sigma)
        return float(peak)

    def extended_problem(self,t,state,mass,durations,previous_plan):
        matrix,target,cap=super().build_problem(t,state,mass,durations)
        count=len(durations);s=self.strategy;n=3*count
        if s.use_bias:
            error=np.r_[np.zeros(6),self.bias]
            scale=np.r_[[1/self.settings.position_scale_m]*3,[1/self.settings.velocity_scale_m_s]*3]
            for i in range(count):
                left=t+i*self.settings.interval_s;h=min(self.settings.interval_s,self.plan.stop_s-left)
                error=self.transition(left,h)@error
                target[6*i:6*i+6]-=error[:6]*scale
        if s.slew_scale_m_s:
            difference=np.eye(n)
            for i in range(3,n):difference[i,i-3]=-1.
            desired=np.zeros(n);desired[:3]=self.previous/s.slew_scale_m_s
            matrix=np.vstack([matrix,difference*cap/s.slew_scale_m_s]);target=np.r_[target,desired]
        if s.sparse_weight:
            norm=np.sqrt(np.sum((previous_plan/cap)**2,axis=1)+s.sparse_epsilon**2)
            weights=np.repeat(np.sqrt(s.sparse_weight/norm),3)
            matrix=np.vstack([matrix,np.diag(weights)]);target=np.r_[target,np.zeros(n)]
        stride=max(1,round(s.minimum_spacing_s/self.settings.interval_s))
        active=np.concatenate([np.arange(3*i,3*i+3) for i in range(0,count,stride)])
        return matrix[:,active],target,cap,active

    def _coast(self,t,reason,status=0,**details):
        self.previous[:]=0;self.previous_epoch=-np.inf if t is None else t;self.last_status=status
        self.last_schedule=np.empty((0,3))
        if status:self.warm[:]=0
        self.diagnostics.append(dict(time_s=t,status=status,reason=reason,**details))
        return np.zeros(3)

    def command(self,t,estimate,mass,*,valid=True):
        if not valid:return self._coast(t,'stale_navigation',2)
        try:self.validate_input(t,estimate,mass)
        except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):return self._coast(None,'invalid_input',1)
        s=self.settings;p=self.strategy;remaining=self.plan.stop_s-t
        count=min(s.horizon_steps,int(np.ceil(remaining/s.interval_s)))
        if count<1:return self._coast(t,'end_of_reference',1)
        if not np.isclose(t-self.previous_epoch,s.interval_s):self.previous[:]=0
        self.bias=np.asarray(estimate['state'][6:9]).copy() if p.use_bias else np.zeros(3)
        state=np.asarray(estimate['state'][:6]);peak=None;overridden=False
        if p.trigger_m or p.minimum_spacing_s:
            try:peak=self.coast_peak(t,estimate,count)
            except (ValueError,TypeError,KeyError,np.linalg.LinAlgError):return self._coast(t,'invalid_covariance',1)
            overridden=peak>p.dwell_override_m
            if p.minimum_spacing_s and t-self.last_burn<p.minimum_spacing_s-1e-6 and not overridden:
                return self._coast(t,'minimum_spacing',coast_peak_m=peak)
            if p.trigger_m and peak<=p.trigger_m:
                return self._coast(t,'predicted_coast_inside_corridor',coast_peak_m=peak)
        warm=self.warm[:count].copy();durations=np.zeros(count);residual=0.
        # Sparse IRLS uses a fixed six reweight/refinement steps. The other
        # strategies retain two impulse/finite-burn refinements.
        solves=6 if p.sparse_weight else 2
        try:
            for iteration in range(solves):
                matrix,target,cap,active=self.extended_problem(t,state,mass,durations,warm)
                h,g,beta=scaled_qp(matrix,target);start=np.clip(warm.ravel()[active]/cap,-1,1)
                if self.solver is None:status,u,residual=float_solve(h,g,start,beta,s.residual_tolerance)
                else:status,u,residual=self.solver.solve(h,g,start,beta,s.residual_tolerance)
                if status:return self._coast(t,'solver_failure',status,residual=residual)
                full=np.zeros(3*count);full[active]=u*cap;warm=full.reshape(count,3)
                durations=np.array([0. if (burn:=self.actuator.command(d,t,mass,interval_s=s.interval_s)) is None else burn.stop_s-burn.start_s for d in warm])
            if np.linalg.norm(warm[0])<p.minimum_command_m_s:return self._coast(t,'small_command',proposed_dv_m_s=warm[0].tolist())
            burn=self.actuator.command(warm[0],t,mass,interval_s=min(s.interval_s,remaining))
            trial=self.model.propagate(state,t,t+min(s.interval_s,remaining),mass,burn,floors=(s.moon_floor_m,s.earth_floor_m))
            if not trial.completed:return self._coast(t,'clearance_screen',32)
        except (ValueError,np.linalg.LinAlgError,RuntimeError) as exc:
            return self._coast(t,'invalid_problem',1,detail=str(exc))
        self.last_status=0;self.last_schedule=warm.copy();self.warm[:]=0
        if count>1:self.warm[:count-1]=warm[1:]
        self.previous=warm[0].copy() if burn is not None else np.zeros(3);self.previous_epoch=t
        if burn is not None:self.last_burn=t
        self.diagnostics.append(dict(time_s=t,status=0,reason='planned_burn',solves=solves,residual=residual,
            coast_peak_m=peak,dwell_overridden=overridden,scheduled_dv_m_s=warm.tolist(),predicted_clearance_m=trial.min_clearance_m))
        return warm[0]


def score_run(row):
    commands=[r for r in row['commands'] if r['burn_s']>0]
    changes=sum(float(np.dot(a['dv'],b['dv']))<0 for a,b in zip(commands[:-1],commands[1:]))
    adjacent=sum(b['time_s']-a['time_s']<=1800.01 and float(np.dot(a['dv'],b['dv']))<0 for a,b in zip(commands[:-1],commands[1:]))
    faults=sum(row['core_status_counts'].get(str(k),0) for k in (1,4,16,32))
    return dict(rms_m=row['dense_steady_rms_m'],peak_m=row['dense_steady_max_m'],dv_m_s=row['dv_m_s'],burns=row['burns'],
                direction_changes=changes,adjacent_reversals=adjacent,solver_or_screen_faults=faults,collision=row['collision'])


def select_strategies(rows):
    expected={(c,s) for c in ('P1_sep','P2_sep') for s in (4101,4102)};summary=[]
    for name in strategies():
        group=[r for r in rows if r['configuration']==name]
        if len(group)!=4 or {(r['case'],r['seed']) for r in group}!=expected or any(r['split']!='training' or r['arithmetic']!='float64' for r in group):
            raise ValueError('Complete training-only strategy grid required')
        metrics=[score_run(r) for r in group]
        feasible=all(not m['collision'] and m['rms_m'] is not None and m['rms_m']<=100 and m['peak_m']<=500 and not m['solver_or_screen_faults'] for m in metrics)
        summary.append(dict(configuration=name,feasible=feasible,worst_rms_m=max(1e99 if m['rms_m'] is None else m['rms_m'] for m in metrics),
            worst_peak_m=max(1e99 if m['peak_m'] is None else m['peak_m'] for m in metrics),mean_dv_m_s=float(np.mean([m['dv_m_s'] for m in metrics])),
            mean_burns=float(np.mean([m['burns'] for m in metrics])),mean_direction_changes=float(np.mean([m['direction_changes'] for m in metrics]))))
    feasible=[r for r in summary if r['feasible']];objectives=('worst_rms_m','mean_dv_m_s','mean_burns')
    front=[r['configuration'] for r in feasible if not any(all(q[k]<=r[k] for k in objectives) and any(q[k]<r[k] for k in objectives) for q in feasible)]
    if not feasible:return dict(primary=None,selected=[],pareto=front,candidates=summary)
    base=next(r for r in summary if r['configuration']=='baseline')
    tight=[r for r in feasible if r['worst_rms_m']<=50 and r['worst_peak_m']<=300]
    pool=tight or feasible
    balanced=min(pool,key=lambda r:r['worst_rms_m']/50+r['mean_dv_m_s']/base['mean_dv_m_s']+r['mean_burns']/base['mean_burns'])
    choices=dict(balanced=balanced['configuration'],tightest=min(feasible,key=lambda r:r['worst_rms_m'])['configuration'],
                 least_fuel=min(feasible,key=lambda r:r['mean_dv_m_s'])['configuration'],fewest_burns=min(pool,key=lambda r:(r['mean_burns'],r['mean_dv_m_s']))['configuration'])
    return dict(primary=balanced['configuration'],choices=choices,selected=list(dict.fromkeys(choices.values())),tight_gate_satisfied=bool(tight),pareto=front,candidates=summary)
