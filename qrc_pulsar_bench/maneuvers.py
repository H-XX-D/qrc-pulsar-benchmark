"""Explicit commanded forces and independent maneuver evidence.

No velocity differences or future reference states are used to infer commands.
Availability epochs distinguish planned inputs from retrospective calibration.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta
import re
import numpy as np


@dataclass(frozen=True)
class FiniteBurn:
    start_s: float
    stop_s: float
    thrust_n: float
    direction_icrf: tuple
    mass_start_kg: float
    mass_flow_kg_s: float
    available_at_s: float
    source: str

    def __post_init__(self):
        values=[self.start_s,self.stop_s,self.thrust_n,self.mass_start_kg,self.mass_flow_kg_s,self.available_at_s]
        direction=np.asarray(self.direction_icrf,dtype=float)
        if not np.all(np.isfinite(values)) or self.stop_s<=self.start_s or self.thrust_n<0 or self.mass_flow_kg_s<0:
            raise ValueError("Invalid finite burn interval or force/mass parameters")
        if self.mass_start_kg-self.mass_flow_kg_s*(self.stop_s-self.start_s)<=0:
            raise ValueError("Burn exhausts spacecraft mass")
        if direction.shape!=(3,) or not np.all(np.isfinite(direction)) or not np.isclose(np.linalg.norm(direction),1,rtol=0,atol=1e-12):
            raise ValueError("Burn direction must be an ICRF unit vector")
        if not self.source: raise ValueError("Independent command source is required")
        object.__setattr__(self,'direction_icrf',tuple(direction))

    def acceleration(self,time_s):
        if not self.start_s<=time_s<self.stop_s: return np.zeros(3)
        mass=self.mass_start_kg-self.mass_flow_kg_s*(time_s-self.start_s)
        return self.thrust_n/mass*np.asarray(self.direction_icrf)


class ManeuverSchedule:
    def __init__(self,burns=()):
        self.burns=tuple(sorted(burns,key=lambda b:b.start_s))
        for left,right in zip(self.burns,self.burns[1:]):
            if left.stop_s>right.start_s:
                raise ValueError("Overlapping burns require an explicitly combined thrust profile")

    def validate_available(self,start,stop,allow_retrospective=False):
        if not allow_retrospective and any(b.stop_s>start and b.start_s<stop and b.available_at_s>start for b in self.burns):
            raise ValueError("Burn information was not available at prediction start")

    def boundaries(self,start,stop):
        return sorted({start,stop,*[t for b in self.burns for t in (b.start_s,b.stop_s) if start<t<stop]})

    def acceleration(self,time_s):
        return sum((b.acceleration(time_s) for b in self.burns),np.zeros(3))


@dataclass(frozen=True)
class ManeuverEvidence:
    start_jd_tdb: float
    stop_jd_tdb: float
    source: str
    description: str
    # Evidence windows flag possible powered flight; they are not force inputs.
    delta_v_vector_available: bool = False

    def __post_init__(self):
        if (not np.all(np.isfinite([self.start_jd_tdb,self.stop_jd_tdb]))
                or self.stop_jd_tdb<=self.start_jd_tdb or not self.source or not self.description):
            raise ValueError("Finite increasing evidence window and source/description required")


def _utc_doy(text):
    from astropy.time import Time
    year,doy,clock=text.split()
    hours,minutes,seconds=clock.split(':')
    utc=datetime(int(year),1,1)+timedelta(days=int(doy)-1,hours=int(hours),minutes=int(minutes),seconds=float(seconds))
    return float(Time(utc,scale='utc').tdb.jd)


def read_lro_small_force_metadata(path,*,source):
    """Read LRO FDF25 metadata without inventing a missing inertial thrust vector.

    These calibration products can use both pre/post-maneuver orbit solutions.
    They are retrospective evidence, not automatically admissible forecast inputs.
    """
    from pathlib import Path
    text=Path(path).read_text()
    def capture(pattern):
        match=re.search(pattern,text)
        if not match: raise ValueError("Missing required FDF25 metadata: "+pattern)
        return match.group(1).strip()
    date=r'(\d{4}\s+\d{3}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?)'
    start=_utc_doy(capture(r'Burn Start \(UTCG\):\s*'+date))
    stop=_utc_doy(capture(r'Burn Stop \(UTCG\):\s*'+date))
    return {
        'event':ManeuverEvidence(start,stop,source,'LRO FDF25 retrospective maneuver calibration'),
        'plan_epoch_jd_tdb':_utc_doy(capture(r'Plan Date \(UTCG\):\s*'+date)),
        'planned_delta_v_mps':float(capture(r'Planned dV \(m/s\):\s*([\d.]+)')),
        'average_spacecraft_mass_kg':float(capture(r'Average SC Mass \(kg\):\s*([\d.]+)')),
        'direction_available':False,
        'classification':'retrospective calibration; not a complete inertial thrust history',
    }
