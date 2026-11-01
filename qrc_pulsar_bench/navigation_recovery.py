"""Causal, bounded hypothesis testing for unannounced velocity changes.

Direct-state fixes are an assumed sensor product. Dynamic consistency cannot
distinguish a real maneuver from a coordinated sensor fault that describes the
same physical trajectory; independent sensing is needed for that distinction.
"""
from collections import deque
from dataclasses import dataclass, replace
import numpy as np
from .adaptive_control import CausalOrbitFilter, square_root_measurement


@dataclass(frozen=True)
class ReacquisitionSettings:
    minimum_jump_m_s: float = .03
    maximum_jump_m_s: float = 2.
    velocity_nis_minimum: float = 25.
    maximum_seed_position_m: float = 5000.
    maximum_track_age_s: float = 10800.
    maximum_fix_gap_s: float = 3600.
    confirmation_span_s: float = 3600.
    confirmation_fixes: int = 3
    confirmation_nis_limit: float = 16.8
    seed_covariance_multiplier: float = 4.
    maximum_position_3sigma_m: float = 200.
    maximum_velocity_3sigma_m_s: float = .02

    def __post_init__(self):
        values=list(vars(self).values())
        if not np.all(np.isfinite(values)) or min(values)<=0:
            raise ValueError('Finite positive reacquisition settings required')
        if (type(self.confirmation_fixes) is not int or self.confirmation_fixes<3
                or self.minimum_jump_m_s>=self.maximum_jump_m_s
                or self.seed_covariance_multiplier<1.):
            raise ValueError('Invalid reacquisition bounds')


class ManeuverAwareOrbitFilter(CausalOrbitFilter):
    """One provisional track, confirmed by subsequent distinct observations.

The first rejected fix seeds a separate filter once a velocity discrepancy is
both large and statistically significant. Subsequent fixes must fit that
propagated state, including known burns. Repeated identical fixes, a velocity
bias without corresponding motion, and isolated outliers cannot confirm it.
No anomaly metadata or truth enters this interface. Memory and work are bounded.
"""
    def __init__(self,*args,reacquisition=ReacquisitionSettings(),**kwargs):
        super().__init__(*args,**kwargs)
        self.reacquisition=reacquisition
        self.candidate=None;self.candidate_start_s=None;self.confirmations=0
        self.candidate_kind=None;self.gap_reacquisition_pending=False
        self.navigation_trusted=True;self.navigation_mode='tracking'
        self.reacquisitions=0;self.candidate_resets=0
        self.events=deque(maxlen=128)

    def _clear_candidate(self):
        self.candidate=None;self.candidate_start_s=None;self.confirmations=0;self.candidate_kind=None

    def observe(self,state_fix,covariance,measurement_s,available_s,commands=()):
        # The base update validates the input and all covariance results before
        # mutating state; duplicate/future fixes must not affect hypothesis state.
        previous_received=self.last_processed_observation_s
        result=super().observe(state_fix,covariance,measurement_s,available_s,commands)
        if result['accepted']:
            self._clear_candidate()
            self.gap_reacquisition_pending=False
            self.navigation_trusted=True;self.navigation_mode='tracking'
            return dict(result,reacquired=False,navigation_mode=self.navigation_mode)
        self.navigation_trusted=False;self.navigation_mode='suspect'
        settings=self.reacquisition
        # A gap in RECEIVED fixes differs from a series of rejected fixes.
        # Once the old track is stale after an actual observation gap, require
        # fresh dynamical evidence instead of anchoring a reset to that prior.
        # Keep this mode through candidate rejection so one bad returning fix
        # cannot permanently prevent reacquisition from later good fixes.
        if np.isfinite(previous_received) and measurement_s-previous_received>settings.maximum_track_age_s:
            self.gap_reacquisition_pending=True
        candidate=self.candidate
        if candidate is not None:
            if measurement_s-candidate.time_s>settings.maximum_fix_gap_s:
                self._clear_candidate();self.candidate_resets+=1
            else:
                confirmation=candidate.observe(state_fix,covariance,measurement_s,available_s,commands)
                if not confirmation['accepted']:
                    self._clear_candidate();self.candidate_resets+=1
                else:
                    self.confirmations+=1
                    position_sigma=3*np.sqrt(np.max(np.linalg.eigvalsh(candidate.covariance[:3,:3])))
                    velocity_sigma=3*np.sqrt(np.max(np.linalg.eigvalsh(candidate.covariance[3:6,3:6])))
                    if (self.confirmations>=settings.confirmation_fixes
                            and measurement_s-self.candidate_start_s>=settings.confirmation_span_s
                            and position_sigma<=settings.maximum_position_3sigma_m
                            and velocity_sigma<=settings.maximum_velocity_3sigma_m_s):
                        # Replace the prior with the independently confirmed
                        # estimate; do not fuse correlated tracks as independent.
                        correction=candidate.state[:6]-self.state[:6]
                        self.state=candidate.state.copy();self._root=candidate._root.copy()
                        self.mass=candidate.mass;self.time_s=candidate.time_s
                        self.last_observation_s=measurement_s
                        self.accepted+=1;self.rejected-=1;self.reacquisitions+=1
                        self.events.append(dict(measurement_s=float(measurement_s),available_s=float(available_s),
                            reason=self.candidate_kind,
                            first_s=float(self.candidate_start_s),confirmations=self.confirmations,
                            state_correction=correction.tolist(),confirmation_nis=confirmation['nis'],
                            position_3sigma_m=float(position_sigma),velocity_3sigma_m_s=float(velocity_sigma)))
                        self._clear_candidate()
                        self.gap_reacquisition_pending=False
                        self.navigation_trusted=True;self.navigation_mode='reacquired'
                        return dict(result,accepted=True,reacquired=True,navigation_mode=self.navigation_mode)
        if self.candidate is None:
            innovation=result['innovation'];velocity=innovation[3:]
            # The main filter has committed only the rejected prediction.
            # Compute this gate from factors too; no return to the unstable
            # covariance-sum solve that caused the original negative NIS.
            velocity_nis,_,_=square_root_measurement(self._root[3:6,:],np.asarray(covariance)[3:,3:],velocity)
            jump=float(np.linalg.norm(velocity))
            recent_velocity_change=(settings.minimum_jump_m_s<=jump<=settings.maximum_jump_m_s
                    and velocity_nis>=settings.velocity_nis_minimum
                    and np.linalg.norm(innovation[:3])<=settings.maximum_seed_position_m
                    and measurement_s-self.last_observation_s<=settings.maximum_track_age_s)
            if self.gap_reacquisition_pending or recent_velocity_change:
                candidate=CausalOrbitFilter(self.model,state_fix,np.asarray(covariance)*settings.seed_covariance_multiplier,
                    self.mass,time_s=measurement_s,
                    settings=replace(self.settings,nis_limit=settings.confirmation_nis_limit))
                # Seeding already consumes this fix. It cannot also confirm it.
                candidate.last_processed_observation_s=float(measurement_s)
                self.candidate=candidate;self.candidate_start_s=float(measurement_s);self.confirmations=1
                self.candidate_kind='observation_gap' if self.gap_reacquisition_pending else 'consistent_velocity_change'
        return dict(result,reacquired=False,navigation_mode=self.navigation_mode)

    def forecast(self,*args,**kwargs):
        result=super().forecast(*args,**kwargs)
        result.update(navigation_trusted=self.navigation_trusted,navigation_mode=self.navigation_mode,
                      candidate_confirmations=self.confirmations)
        return result
