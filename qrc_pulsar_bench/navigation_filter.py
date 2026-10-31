"""Sequential orbit/clock EKF for supplied plane-wave pseudorange observations.

State: ICRF r (m), v (m/s), receiver clock bias (m), drift (m/s). Observation
contract: z_i = direction_i dot r + bias + noise. Directions/sign/phase offsets
must be supplied by the observation adapter. This is not a raw pulsar-TOA model:
dispersion, phase ambiguity, relativistic delays and ephemeris errors are absent.
"""
import numpy as np
from scipy.integrate import solve_ivp


class OrbitClockEKF:
    def __init__(self,state,covariance,acceleration,*,time_s=0.0,
                 acceleration_noise_psd=0.0,clock_drift_noise_psd=0.0,state_acceleration=None):
        self.state=np.array(state,dtype=float,copy=True)
        self.covariance=np.array(covariance,dtype=float,copy=True)
        if self.state.shape!=(8,) or self.covariance.shape!=(8,8) or not np.all(np.isfinite(self.state)):
            raise ValueError("Expected finite eight-state orbit/clock initial condition")
        self._validate_covariance(self.covariance)
        if not np.all(np.isfinite([time_s,acceleration_noise_psd,clock_drift_noise_psd])) or min(acceleration_noise_psd,clock_drift_noise_psd)<0:
            raise ValueError("Invalid time/process spectral densities")
        self.acceleration=acceleration; self.time_s=float(time_s)
        # Optional full a(r,v,t) replaces the position-only callback, allowing
        # EIH forces and their velocity Jacobian without adding them twice.
        self.state_acceleration=state_acceleration
        self.process=np.zeros((8,8)); self.process[3:6,3:6]=np.eye(3)*acceleration_noise_psd
        self.process[7,7]=clock_drift_noise_psd

    @staticmethod
    def _validate_covariance(matrix):
        if not np.all(np.isfinite(matrix)) or not np.allclose(matrix,matrix.T,rtol=0,atol=1e-8):
            raise ValueError("Covariance must be finite and symmetric")
        if np.min(np.linalg.eigvalsh(matrix))<-1e-10:
            raise ValueError("Covariance must be positive semidefinite")

    def predict_to(self,time_s,max_step_s=120.0):
        if not np.isfinite(time_s) or time_s<self.time_s or not np.isfinite(max_step_s) or max_step_s<=0:
            raise ValueError("Prediction time must be monotonic; step must be positive")
        if time_s==self.time_s: return
        def rhs(t,y):
            state=y[:8]; p=y[8:].reshape(8,8)
            force=lambda r,v:np.asarray(self.acceleration(r,t) if self.state_acceleration is None else self.state_acceleration(r,v,t),dtype=float)
            a=force(state[:3],state[3:6])
            if a.shape!=(3,) or not np.all(np.isfinite(a)): raise ValueError("Invalid force output")
            A=np.zeros((8,8)); A[:3,3:6]=np.eye(3); A[6,7]=1
            # Spatial finite differences of the *force*, never trajectory data.
            h=10.0
            for j,e in enumerate(np.eye(3)):
                A[3:6,j]=(force(state[:3]+h*e,state[3:6])-force(state[:3]-h*e,state[3:6]))/(2*h)
                if self.state_acceleration is not None:
                    dv=1e-3
                    A[3:6,3+j]=(force(state[:3],state[3:6]+dv*e)-force(state[:3],state[3:6]-dv*e))/(2*dv)
            derivative=np.r_[state[3:6],a,state[7],0]
            return np.r_[derivative,(A@p+p@A.T+self.process).ravel()]
        initial=np.r_[self.state,self.covariance.ravel()]
        result=solve_ivp(rhs,(self.time_s,time_s),initial,method='DOP853',rtol=1e-10,atol=1e-8,max_step=max_step_s)
        if not result.success: raise RuntimeError(result.message)
        state=result.y[:8,-1]; covariance=result.y[8:,-1].reshape(8,8)
        covariance=.5*(covariance+covariance.T)
        self._validate_covariance(covariance)
        self.state=state; self.covariance=covariance; self.time_s=float(time_s)

    def observe(self,time_s,pseudorange_m,directions_icrf,noise_covariance,*,nis_limit=None):
        z=np.asarray(pseudorange_m,dtype=float); directions=np.asarray(directions_icrf,dtype=float)
        R=np.asarray(noise_covariance,dtype=float)
        if z.ndim!=1 or len(z)==0 or directions.shape!=(len(z),3) or R.shape!=(len(z),len(z)):
            raise ValueError("Observation dimensions do not match")
        if not np.all(np.isfinite(z)) or not np.all(np.isfinite(directions)) or not np.allclose(np.linalg.norm(directions,axis=1),1,atol=1e-12,rtol=0):
            raise ValueError("Finite pseudoranges and unit directions required")
        self._validate_covariance(R)
        if np.min(np.linalg.eigvalsh(R))<=0: raise ValueError("Measurement covariance must be positive definite")
        if nis_limit is not None and (not np.isfinite(nis_limit) or nis_limit<=0): raise ValueError("Positive NIS gate required")
        self.predict_to(time_s)
        H=np.zeros((len(z),8)); H[:,:3]=directions; H[:,6]=1
        innovation=z-H@self.state; S=H@self.covariance@H.T+R
        nis=float(innovation@np.linalg.solve(S,innovation))
        accepted=nis_limit is None or nis<=nis_limit
        if accepted:
            gain=np.linalg.solve(S,H@self.covariance).T
            self.state=self.state+gain@innovation
            I_KH=np.eye(8)-gain@H
            # Joseph form preserves covariance PSD under floating-point error.
            self.covariance=I_KH@self.covariance@I_KH.T+gain@R@gain.T
            self.covariance=.5*(self.covariance+self.covariance.T)
        return {'time_s':self.time_s,'accepted':accepted,'normalized_innovation_squared':nis,
                'state':self.state.copy(),'covariance':self.covariance.copy()}
