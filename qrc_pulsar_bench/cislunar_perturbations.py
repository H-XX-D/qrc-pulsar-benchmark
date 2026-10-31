"""Additional sourced perturbations and explicitly labelled lunar-tide scenarios.

SI, Moon-relative ICRF vectors and TDB seconds. The first-PN calculation is
formed in barycentric coordinates and subtracts the lunar-origin correction;
adding an isolated Schwarzschild term for each body would omit cross terms.
Lunar tide conventions must be supplied explicitly; neither scenario below
claims to resolve a gravity product's undocumented permanent-tide convention.
"""
from dataclasses import dataclass
import re
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicHermiteSpline
from .cislunar_forces import GM_EARTH, GM_MOON, GM_SUN, R_EARTH, j2_acceleration
from .constants import c


def read_gm_kernel(path):
    """Scalar GM records, including Fortran D exponents, converted km³ to m³."""
    text=Path(path).read_text()
    return {int(k):float(val.replace('D','E').replace('d','e'))*1e9 for k,val in
            re.findall(r'BODY(\d+)_GM\s*=\s*\(\s*([\d.EeDd+\-]+)\s*\)',text)}


def induced_tide_tensor(body_position, body_gm, radius_m, love_k2):
    """Trace-free Q with induced degree-two potential U = r.T Q r / |r|^5.

    U = k2 GM_body R^5 / (d^3 r^3) P2(cos(angle)); instantaneous elastic tide.
    Rotation of Q, rather than only the tidal axis, also permits a specified
    permanent/reference tensor to be subtracted consistently in body axes.
    """
    body = np.asarray(body_position, dtype=float)
    d = np.linalg.norm(body)
    if (body.shape != (3,) or not np.all(np.isfinite(body)) or d == 0
            or not np.all(np.isfinite([body_gm,radius_m,love_k2]))
            or body_gm < 0 or radius_m <= 0 or love_k2 < 0):
        raise ValueError("Invalid tidal body, GM, radius or Love number")
    u = body/d
    return .5*love_k2*body_gm*radius_m**5/d**3 * (3*np.outer(u,u)-np.eye(3))


def tensor_acceleration(r, tensor):
    r = np.asarray(r,dtype=float)
    tensor = np.asarray(tensor,dtype=float)
    radius = np.linalg.norm(r)
    if (r.shape != (3,) or tensor.shape != (3,3) or radius == 0
            or not np.all(np.isfinite(r)) or not np.all(np.isfinite(tensor))
            or not np.allclose(tensor,tensor.T,rtol=1e-13,atol=0)):
        raise ValueError("Finite nonzero position and symmetric tensor required")
    return 2*(tensor@r)/radius**5 - 5*(r@tensor@r)*r/radius**7


def eih_vectorized(positions, velocities, gm):
    """Order-reduced EIH correction, beta=gamma=1; GM inputs instead of masses.

    Vectorized counterpart of relativity.eih_correction; returns only 1PN.
    Zero-mass test particles are allowed. All velocities must be barycentric.
    Source: JPL Standish/Williams, Solar System Ephemerides, Eq. 8-1.
    """
    x,v,gm = (np.asarray(a,dtype=float) for a in (positions,velocities,gm))
    n = len(gm)
    if (gm.shape != (n,) or x.shape != (n,3) or v.shape != (n,3)
            or not all(np.all(np.isfinite(a)) for a in (x,v,gm)) or np.any(gm<0)):
        raise ValueError("Invalid barycentric state or GM")
    d = x[:,None,:]-x[None,:,:]
    radius = np.linalg.norm(d,axis=2)
    np.fill_diagonal(radius,np.inf)
    if np.any(radius == 0):
        raise ValueError("Point-mass collision")
    inv3 = radius**-3
    potential = (1/radius)@gm
    newton = -np.sum((inv3*gm[None,:])[:,:,None]*d,axis=1)
    vv = np.sum(v*v,axis=1)
    radial = (4*potential[:,None]+potential[None,:]-vv[:,None]-2*vv[None,:]
              +4*(v@v.T)+1.5*(np.sum(d*v[None,:,:],axis=2)/radius)**2
              +.5*np.sum(d*newton[None,:,:],axis=2))
    velocity = np.sum(d*(4*v[:,None,:]-3*v[None,:,:]),axis=2)[:,:,None]*(v[:,None,:]-v[None,:,:])
    terms = (radial[:,:,None]*d+velocity)*inv3[:,:,None] + 3.5*newton[None,:,:]/radius[:,:,None]
    return np.sum(gm[None,:,None]*terms,axis=1)/c**2


class BarycentricEnvironment:
    """Local planetary kernel, fixed-grid Hermite preload, no spacecraft data.

    Planetary-system barycentres use matching system GMs from gm_de440.tpc.
    The Moon, Earth and Sun occupy indices 0,1,2. Earth J2 uses the IAU 2006/2000A
    celestial pole; polar motion and time-dependent Earth harmonics are omitted.
    """
    bodies = ((301,((0,3),(3,301))), (399,((0,3),(3,399))), (10,((0,10),)),
              (1,((0,1),)), (2,((0,2),)), (4,((0,4),)), (5,((0,5),)),
              (6,((0,6),)), (7,((0,7),)), (8,((0,8),)), (9,((0,9),)))

    def __init__(self,epoch_jd_tdb,duration_s,kernel_path,gm_path,grid_step_s=3600.):
        from jplephem.spk import SPK
        from astropy.time import Time,TimeDelta
        import erfa
        if not np.all(np.isfinite([epoch_jd_tdb,duration_s,grid_step_s])) or min(duration_s,grid_step_s)<=0:
            raise ValueError("Invalid ephemeris interval")
        self.duration_s=duration_s
        grid=np.linspace(0,duration_s,max(2,int(np.ceil(duration_s/grid_step_s))+1))
        times=Time(epoch_jd_tdb,format='jd',scale='tdb')+TimeDelta(grid,format='sec')
        values=read_gm_kernel(gm_path)
        self.gm=np.array([values[code] for code,_ in self.bodies])
        positions=[];velocities=[]
        with SPK.open(str(kernel_path)) as kernel:
            for _,chain in self.bodies:
                pairs=[kernel[a,b].compute_and_differentiate(times.tdb.jd1,times.tdb.jd2) for a,b in chain]
                positions.append(sum(p for p,v in pairs).T*1e3)
                velocities.append(sum(v for p,v in pairs).T*1e3/86400)
        self.spline=CubicHermiteSpline(grid,np.stack(positions,axis=1),np.stack(velocities,axis=1),extrapolate=False)
        tt=times.tt
        poles=erfa.pnm06a(tt.jd1,tt.jd2)[:,2,:]
        self.pole_spline=CubicHermiteSpline(grid,poles,np.gradient(poles,grid,axis=0),extrapolate=False)
        self.source=str(Path(kernel_path).resolve())

    def __call__(self,t):
        if not np.isfinite(t) or not 0<=t<=self.duration_s:
            raise ValueError("Epoch outside planetary preload")
        return self.spline(t),self.spline(t,1)

    def earth_pole(self,t):
        self(t)  # coverage check
        pole=self.pole_spline(t)
        return pole/np.linalg.norm(pole)


@dataclass(frozen=True)
class PerturbationOptions:
    earth_j2: bool = False
    other_planets: bool = False
    first_pn: bool = False
    tide_convention: str = 'none'  # none, full_elastic, epoch_subtracted
    love_k2: float = .024116
    earth_j2_value: float = np.sqrt(5)*.48416948e-3  # IERS conventional normalized C20

    def __post_init__(self):
        if self.tide_convention not in ('none','full_elastic','epoch_subtracted'):
            raise ValueError("Explicit supported tide convention required")
        if not np.all(np.isfinite([self.love_k2,self.earth_j2_value])) or min(self.love_k2,self.earth_j2_value)<0:
            raise ValueError("Invalid quadrupole parameters")


class CislunarPerturbations:
    """Callable extra(r,v,t), added once to the existing Newtonian force model.

    first_pn includes all configured solar-system point masses, subtracting the
    lunar correction. Multipolar relativistic forces and dissipation are absent.
    epoch_subtracted removes the elastic tensor at t0 in lunar body axes; this
    is a sensitivity convention, not a claim about GRGM900C's permanent tide.
    """
    def __init__(self,environment,orientation,options=PerturbationOptions(),radius_m=1738e3):
        self.environment=environment; self.orientation=orientation; self.options=options
        self.radius_m=radius_m
        x,_=environment(0)
        q=self._tidal_tensor(x)
        rotation=orientation(0)
        self.reference_body_tensor=rotation.T@q@rotation

    def _tidal_tensor(self,x):
        return sum((induced_tide_tensor(x[j]-x[0],self.environment.gm[j],self.radius_m,self.options.love_k2)
                    for j in (1,2)),np.zeros((3,3)))

    def components(self,r,v,t):
        x,vel=self.environment(t)
        r=np.asarray(r,dtype=float); v=np.asarray(v,dtype=float)
        terms={}
        if self.options.earth_j2:
            earth=x[1]-x[0]; pole=self.environment.earth_pole(t)
            terms['earth_j2']=j2_acceleration(r-earth,GM_EARTH,R_EARTH,self.options.earth_j2_value,pole)-j2_acceleration(-earth,GM_EARTH,R_EARTH,self.options.earth_j2_value,pole)
        if self.options.other_planets:
            bodies=x[3:]-x[0]
            d=bodies-r
            terms['other_planets']=np.sum(self.environment.gm[3:,None]*(d/np.linalg.norm(d,axis=1)[:,None]**3-bodies/np.linalg.norm(bodies,axis=1)[:,None]**3),axis=0)
        if self.options.first_pn:
            corrections=eih_vectorized(np.vstack((x,x[0]+r)),np.vstack((vel,vel[0]+v)),np.r_[self.environment.gm,0.])
            terms['first_pn']=corrections[-1]-corrections[0]
        if self.options.tide_convention!='none':
            q=self._tidal_tensor(x)
            if self.options.tide_convention=='epoch_subtracted':
                rotation=self.orientation(t)
                q-=rotation@self.reference_body_tensor@rotation.T
            q=.5*(q+q.T)
            terms['lunar_elastic_tide']=tensor_acceleration(r,q)
        return terms

    def __call__(self,r,v,t):
        return sum(self.components(r,v,t).values(),np.zeros(3))
