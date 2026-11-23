"""Bounded recovery after continuously received but rejected measurements."""
from dataclasses import replace
import numpy as np
from .navigation_recovery import ManeuverAwareOrbitFilter


class LossOfTrackFilter(ManeuverAwareOrbitFilter):
    """A stale accepted track is distinct from an absence of received packets.

    Ordinary maneuver recovery is unchanged. After three hours of consecutive
    rejection, a new candidate must satisfy FIVE fixes spanning two hours.
    This remains consistency evidence, not authentication of physical motion.
    Unphysical candidate state jumps inconsistent with velocity change and
    elapsed time are rejected as sensor bias.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.normal_reacquisition=self.reacquisition
        self.rejection_start_s=None
        self.persistent_recovery=False

    def observe(self,z,r,measurement_s,available_s,commands=()):
        # Validate/update first: malformed inputs cannot arm a reset.
        candidate=self.candidate
        if candidate is not None and getattr(self,'candidate_kind',None)=='persistent_rejection':
            pos_diff=float(np.linalg.norm(candidate.state[:3]-self.state[:3]))
            vel_diff=float(np.linalg.norm(candidate.state[3:6]-self.state[3:6]))
            dt=max(1.,float(measurement_s-(self.rejection_start_s or measurement_s)))
            max_phys_pos=1.8*(vel_diff+0.015)*dt+150.
            if pos_diff>max_phys_pos:
                self._clear_candidate()
                self.candidate_resets+=1
        result=super().observe(z,r,measurement_s,available_s,commands)
        if result['accepted']:
            self.rejection_start_s=None
            self.persistent_recovery=False
            self.reacquisition=self.normal_reacquisition
        else:
            if self.rejection_start_s is None:self.rejection_start_s=float(measurement_s)
            if (not self.persistent_recovery
                    and measurement_s-self.rejection_start_s>=10800.
                    and measurement_s-self.last_observation_s>=10800.):
                self._clear_candidate()
                self.persistent_recovery=True
                self.reacquisition=replace(self.normal_reacquisition,
                    confirmation_fixes=5,confirmation_span_s=7200.)
            if self.persistent_recovery:
                # Arm for the NEXT distinct measurement, never reuse this fix.
                self.gap_reacquisition_pending=True
                if self.candidate is not None:self.candidate_kind='persistent_rejection'
        return result
