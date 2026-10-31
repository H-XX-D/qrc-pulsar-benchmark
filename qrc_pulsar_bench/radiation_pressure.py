"""Attitude-dependent flat-plate SRP with explicit optical/mass inputs.

No plate self-shadowing, thermal recoil or lunar albedo. Supply body-to-ICRF
attitude independently; archived fitted spacecraft states are not inputs.
"""
from dataclasses import dataclass
import numpy as np
from .cislunar_forces import AU,SOLAR_PRESSURE_1AU,sunlight_fraction


@dataclass(frozen=True)
class OpticalPlate:
    area_m2: float
    normal_body: tuple
    specular: float = 0.0
    diffuse: float = 0.0

    def __post_init__(self):
        n=np.asarray(self.normal_body,dtype=float)
        if not np.all(np.isfinite([self.area_m2,self.specular,self.diffuse])) or self.area_m2<=0 or min(self.specular,self.diffuse)<0 or self.specular+self.diffuse>1:
            raise ValueError("Invalid area or optical energy fractions")
        if n.shape!=(3,) or not np.all(np.isfinite(n)) or not np.isclose(np.linalg.norm(n),1,rtol=0,atol=1e-12):
            raise ValueError("Plate normal must be a unit body-frame vector")
        object.__setattr__(self,'normal_body',tuple(n))


class PlateRadiationPressure:
    def __init__(self,mass_kg,plates,body_to_icrf,*,source,available_at_s):
        if not np.isfinite(mass_kg) or mass_kg<=0 or not source or not np.isfinite(available_at_s):
            raise ValueError("Positive mass and explicit source/availability required")
        self.mass_kg=float(mass_kg); self.plates=tuple(plates); self.body_to_icrf=body_to_icrf
        self.source=source; self.available_at_s=float(available_at_s)

    def __call__(self,r,earth,sun,time_s):
        rotation=np.asarray(self.body_to_icrf(time_s),dtype=float)
        if rotation.shape!=(3,3) or not np.all(np.isfinite(rotation)) or not np.allclose(rotation.T@rotation,np.eye(3),rtol=0,atol=1e-10) or np.linalg.det(rotation)<0:
            raise ValueError("Attitude must be a proper body-to-ICRF rotation")
        to_sun=np.asarray(sun)-r; distance=np.linalg.norm(to_sun); direction=to_sun/distance
        pressure=SOLAR_PRESSURE_1AU*(AU/distance)**2*sunlight_fraction(r,earth,sun)
        force=np.zeros(3)
        for plate in self.plates:
            normal=rotation@plate.normal_body
            incidence=max(0.,float(np.dot(normal,direction)))
            force-=pressure*plate.area_m2*incidence*((1-plate.specular)*direction
                +2*(plate.specular*incidence+plate.diffuse/3)*normal)
        return force/self.mass_kg
