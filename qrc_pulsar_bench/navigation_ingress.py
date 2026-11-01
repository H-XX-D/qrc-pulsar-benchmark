"""Reject malformed sensor packets without masking numerical filter failures."""
import numpy as np
from .adaptive_control import validate_covariance


class NavigationIngress:
    def __init__(self,filter_):
        self.filter=filter_;self.input_healthy=True;self.invalid_packets=0

    def observe(self,fix,covariance,measurement_s,available_s,commands=()):
        reason=None
        try:
            z=np.asarray(fix,dtype=float);r=np.asarray(covariance,dtype=float)
            if z.shape!=(6,) or not np.all(np.isfinite(z)):raise ValueError('invalid_state_fix')
            if r.shape!=(6,6):raise ValueError('invalid_measurement_covariance')
            validate_covariance(r)
            if np.linalg.eigvalsh(r)[0]<=0:raise ValueError('invalid_measurement_covariance')
            if not np.all(np.isfinite([measurement_s,available_s])):raise ValueError('invalid_timestamp')
            if measurement_s>available_s:raise ValueError('future_timestamp')
            if measurement_s<=self.filter.last_processed_observation_s:raise ValueError('replayed_or_reversed_timestamp')
        except (ValueError,TypeError) as error:reason=str(error)
        if reason is not None:
            self.input_healthy=False;self.invalid_packets+=1
            return dict(accepted=False,nis=None,reacquired=False,input_error=reason,navigation_mode='invalid_packet')
        # Deliberately outside the input-validation catch: a broken propagation
        # or covariance calculation must surface as a numerical failure.
        result=self.filter.observe(z,r,measurement_s,available_s,commands)
        self.input_healthy=True
        return dict(result,input_error=None)

    def forecast(self,*args,**kwargs):
        result=self.filter.forecast(*args,**kwargs)
        result['navigation_trusted']=bool(self.input_healthy and result.get('navigation_trusted',True))
        if not self.input_healthy:result['navigation_mode']='invalid_packet'
        return result
