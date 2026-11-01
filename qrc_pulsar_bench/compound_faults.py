"""Predeclared plant/sensor/link faults; never supplied to the navigator or MPC."""
from dataclasses import dataclass
import numpy as np

HOUR=3600.
KINDS={'impulse','acceleration','acceleration_ramp','thrust_gain','pointing',
       'uncommanded_thrust','sensor_bias','sensor_noise','sensor_dropout',
       'sensor_replay','sensor_nan','sensor_clock','ground_blackout'}


@dataclass(frozen=True)
class Fault:
    kind: str
    start_s: float
    end_s: float
    value: float = 0.
    second_value: float = 0.

    def __post_init__(self):
        if self.kind not in KINDS:raise ValueError('Unknown fault kind')
        if not np.all(np.isfinite([self.start_s,self.end_s,self.value,self.second_value])):
            raise ValueError('Finite fault parameters required')
        if self.start_s<0 or self.end_s<self.start_s or self.start_s%1800 or self.end_s%1800:
            raise ValueError('Fault boundaries must align to1800s grid')
        if self.kind=='impulse':
            if self.end_s!=self.start_s:raise ValueError('Impulse has a single epoch')
        elif self.end_s==self.start_s:raise ValueError('Finite duration required')
        if self.kind!='sensor_clock' and self.value<0:raise ValueError('Negative fault magnitude')
        if self.kind=='sensor_noise' and self.value<1:raise ValueError('Noise multiplier must be at least one')

    def active(self,t):return self.start_s<=t<self.end_s


@dataclass(frozen=True)
class FaultCase:
    name: str
    category: str
    description: str
    events: tuple

    @property
    def onset_s(self):return min((e.start_s for e in self.events),default=12*HOUR)
    @property
    def end_s(self):return max((e.end_s for e in self.events),default=12*HOUR)


def suite():
    def f(kind,start,end,value=0.,second=0.):return Fault(kind,start*HOUR,end*HOUR,value,second)
    return [
        FaultCase('01_ground_blackout','single','48h ground-link loss with onboard fixes retained',(f('ground_blackout',12,60),)),
        FaultCase('02_future_sensor_clock','single','Fix timestamps120s ahead for6h',(f('sensor_clock',12,18,120),)),
        FaultCase('03_replayed_navigation','single','Replay the last delivered packet for6h',(f('sensor_replay',12,18),)),
        FaultCase('04_nonfinite_fix','single','One delivered fix contains a NaN velocity',(f('sensor_nan',12,12.5),)),
        FaultCase('05_unreported_noise','single','Fivefold sensor noise for12h; reported covariance unchanged',(f('sensor_noise',12,24,5),)),
        FaultCase('06_position_bias','single','1km position-only sensor bias for6h',(f('sensor_bias',12,18,1000),)),
        FaultCase('07_temporary_thrust_loss','single','Zero delivered commanded thrust and flow for12h',(f('thrust_gain',12,24,0),)),
        FaultCase('08_thrust_overdelivery','single','150% delivered thrust and flow for12h',(f('thrust_gain',12,24,1.5),)),
        FaultCase('09_pointing_error','single','60degree command-direction rotation for12h',(f('pointing',12,24,np.pi/3),)),
        FaultCase('10_ramping_force','single','Unknown acceleration ramps0 to2e-6m/s² over12h',(f('acceleration_ramp',12,24,2e-6),)),
        FaultCase('11_kick_and_weak_thrust','concurrent','0.1m/s kick plus50% thrust for12h',
            (f('impulse',12,12,.1),f('thrust_gain',12,24,.5))),
        FaultCase('12_kick_and_sensor_loss','concurrent','0.2m/s kick plus3h navigation outage',
            (f('impulse',12,12,.2),f('sensor_dropout',12,15))),
        FaultCase('13_force_and_sensor_bias','concurrent','12h force error plus6h biased position/velocity',
            (f('acceleration',12,24,1e-6),f('sensor_bias',12,18,300,.008))),
        FaultCase('14_pointing_weak_thrust_noise','concurrent','Mispointing,70% thrust loss and unreported noise',
            (f('pointing',12,24,np.pi/6),f('thrust_gain',12,24,.3),f('sensor_noise',12,18,5))),
        FaultCase('15_stuck_thrust_blackout','concurrent','Uncommanded0.2N thrust6h plus3h sensor and48h ground outages',
            (f('uncommanded_thrust',12,18,.2),f('sensor_dropout',12,15),f('ground_blackout',12,60))),
        FaultCase('16_three_velocity_steps','consecutive','Independent0.03/0.05/0.08m/s kicks12h apart',
            (f('impulse',12,12,.03),f('impulse',24,24,.05),f('impulse',36,36,.08))),
        FaultCase('17_force_bias_then_actuator','consecutive','Force pulse, sensor bias, then12h thrust loss',
            (f('acceleration',12,18,1e-6),f('sensor_bias',24,30,200,.006),f('thrust_gain',36,48,0))),
        FaultCase('18_bad_sensor_return_then_kick','consecutive','6h outage,6h noisy return, then0.2m/s kick',
            (f('sensor_dropout',12,18),f('sensor_noise',18,24,4),f('impulse',30,30,.2))),
        FaultCase('19_thrust_kick_clock','consecutive','Thrust loss, subsequent kick, then malformed sensor clock',
            (f('thrust_gain',12,24,0),f('impulse',30,30,.1),f('sensor_clock',36,39,120))),
        FaultCase('20_cascading_failures','consecutive','Kick, outage, force pulse, weak thrust, final bad packet during recovery',
            (f('impulse',12,12,.15),f('sensor_dropout',18,24),f('acceleration',28,34,1e-6),
             f('thrust_gain',40,52,.5),f('sensor_nan',56,56.5))),
    ]
