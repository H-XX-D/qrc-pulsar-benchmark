"""Host bridge and independent integer oracle for a bounded FPGA PID candidate.

Hardware is unspecified. Host bit-exact arithmetic tests are not RTL, synthesis,
timing or power measurements. The core consumes normalized state errors and a
model-derived feedforward impulse; force propagation/navigation remain separate.
"""
import ctypes
from dataclasses import dataclass
from pathlib import Path
import subprocess
import numpy as np


POSITION_SCALE = 4096.  # metres
VELOCITY_SCALE = .25   # metres/second
DV_SCALE = .25         # metres/second


@dataclass(frozen=True)
class PIDSettings:
    interval_s: float = 1800.
    response_s: float = 21600.
    integral_strength: float = .1
    derivative_filter_s: float = 1800.
    max_dv_m_s: float = .15
    integral_limit_m_s: float = .03

    def __post_init__(self):
        v=np.asarray(list(vars(self).values()))
        if not np.all(np.isfinite(v)) or np.any(v<0) or min(self.interval_s,self.response_s,self.max_dv_m_s)<=0:
            raise ValueError('Invalid PID settings')
        if self.integral_limit_m_s>self.max_dv_m_s/np.sqrt(3):raise ValueError('Integral limit exceeds axis limit')

    def normalized(self):
        h,tau=self.interval_s,self.response_s
        # Integral state is its contribution to the current impulse command.
        return np.array([h*POSITION_SCALE/(tau*tau*DV_SCALE),
                         2*h*VELOCITY_SCALE/(tau*DV_SCALE),
                         self.integral_strength*h*h*POSITION_SCALE/(tau**3*DV_SCALE),
                         h/(self.derivative_filter_s+h),
                         self.max_dv_m_s/(np.sqrt(3)*DV_SCALE),self.integral_limit_m_s/DV_SCALE])


class FixedFormat:
    def __init__(self,bits=32,fraction=24):
        if not 16<=bits<=32 or not 0<fraction<bits-1:raise ValueError('Invalid fixed-point format')
        self.bits,self.fraction,self.scale=bits,fraction,1<<fraction
        self.lo,self.hi=-(1<<(bits-1)),(1<<(bits-1))-1

    def encode(self,x):
        x=np.asarray(x,dtype=float)
        if not np.all(np.isfinite(x)):raise ValueError('Nonfinite fixed-point input')
        q=np.sign(x)*np.floor(abs(x)*self.scale+.5)
        if np.any(q<self.lo) or np.any(q>self.hi):raise ValueError('Fixed-point input outside representable range')
        return np.ascontiguousarray(q,dtype=np.int32)

    def decode(self,q):return np.asarray(q,dtype=float)/self.scale

    def oracle(self,error,velocity,feedforward,config,state,valid=True):
        """Python unlimited integers implement the written arithmetic contract."""
        c,s=[int(x) for x in config],[int(x) for x in state]
        zero=np.zeros(3,dtype=np.int32)
        if any(x<self.lo or x>self.hi for x in c+s) or min(c[:4])<0 or c[3]>self.scale or c[4]<=0 or not 0<=c[5]<=c[4]:return 1,np.array(s,dtype=np.int32),zero
        if not valid:return 2,np.array(s,dtype=np.int32),zero
        if any(int(x)<self.lo or int(x)>self.hi for a in (error,velocity,feedforward) for x in a):return 1,np.array(s,dtype=np.int32),zero
        fault=False
        def sat(x):
            nonlocal fault
            if x<self.lo or x>self.hi:fault=True
            return max(self.lo,min(self.hi,x))
        def mul(a,b):
            x=int(a)*int(b);y=(abs(x)+(1<<(self.fraction-1)))>>self.fraction
            return sat(-y if x<0 else y)
        def limit(x,b):return max(-b,min(b,x))
        next_state=s.copy();out=[];status=0
        for j in range(3):
            derivative=sat(s[j+3]+mul(c[3],sat(int(velocity[j])-s[j+3])))
            increment=mul(c[2],int(error[j]));integral=limit(sat(s[j]+increment),c[5])
            pd=sat(sat(int(feedforward[j])-mul(c[0],int(error[j])))-mul(c[1],derivative))
            requested=sat(pd-integral)
            if (requested>c[4] and increment<0) or (requested<-c[4] and increment>0):
                integral=s[j];requested=sat(pd-integral)
            command=limit(requested,c[4]);status|=8 if command!=requested else 0
            next_state[j]=integral;next_state[j+3]=derivative;out.append(command)
        if fault:return 4,np.array(s,dtype=np.int32),zero
        return status,np.array(next_state,dtype=np.int32),np.array(out,dtype=np.int32)


def compile_core(output,bits=32,fraction=24):
    source=Path(__file__).resolve().parents[1]/'hardware'
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run(['clang++','-std=c++11','-O2','-shared','-fPIC','-Wno-unknown-pragmas',
        f'-DQRC_PID_BITS={bits}',f'-DQRC_PID_FRAC={fraction}',str(source/'orbit_pid_core.cpp'),'-o',str(output)],check=True)
    return output


class FixedPID:
    def __init__(self,library,settings=PIDSettings(),*,bits=32,fraction=24):
        self.settings=settings;self.format=FixedFormat(bits,fraction)
        self.config=self.format.encode(settings.normalized());self.state=np.zeros(6,dtype=np.int32)
        self.library=ctypes.CDLL(str(Path(library).resolve()));self.function=self.library.orbit_pid_step
        pointer=np.ctypeslib.ndpointer(dtype=np.int32,ndim=1,flags='C_CONTIGUOUS')
        self.function.argtypes=[pointer,pointer,pointer,pointer,ctypes.c_int,pointer,pointer]
        self.function.restype=ctypes.c_int;self.last_status=0

    def step(self,error_m,velocity_error_m_s,feedforward_dv_m_s,*,valid=True):
        arrays=[np.asarray(x,dtype=float) for x in (error_m,velocity_error_m_s,feedforward_dv_m_s)]
        if any(x.shape!=(3,) for x in arrays):raise ValueError('Three axes required')
        if not valid:self.last_status=2;return np.zeros(3)
        try:
            e,v,ff=[self.format.encode(x/scale) for x,scale in zip(arrays,(POSITION_SCALE,VELOCITY_SCALE,DV_SCALE))]
        except ValueError:self.last_status=1;return np.zeros(3)
        out=np.zeros(3,dtype=np.int32)
        self.last_status=int(self.function(e,v,ff,self.config,int(valid),self.state,out))
        return self.format.decode(out)*DV_SCALE


class FloatPID:
    def __init__(self,settings=PIDSettings()):
        self.settings=settings;self.config=settings.normalized();self.state=np.zeros(6);self.last_status=0

    def step(self,error_m,velocity_error_m_s,feedforward_dv_m_s,*,valid=True):
        if not valid:self.last_status=2;return np.zeros(3)
        e=np.asarray(error_m)/POSITION_SCALE;v=np.asarray(velocity_error_m_s)/VELOCITY_SCALE;ff=np.asarray(feedforward_dv_m_s)/DV_SCALE
        if any(x.shape!=(3,) for x in (e,v,ff)) or not np.all(np.isfinite(np.r_[e,v,ff])):self.last_status=1;return np.zeros(3)
        kp,kd,ki,alpha,cap,icap=self.config
        derivative=self.state[3:]+alpha*(v-self.state[3:]);increment=ki*e
        integral=np.clip(self.state[:3]+increment,-icap,icap)
        pd=ff-kp*e-kd*derivative;requested=pd-integral
        freeze=((requested>cap)&(increment<0))|((requested<-cap)&(increment>0))
        integral=np.where(freeze,self.state[:3],integral);requested=pd-integral
        command=np.clip(requested,-cap,cap);self.last_status=8 if np.any(command!=requested) else 0
        self.state=np.r_[integral,derivative]
        return command*DV_SCALE
