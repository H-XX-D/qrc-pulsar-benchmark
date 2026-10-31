"""Normalized lunar gravity with explicit model/frame provenance.

GRGM900C uses its own GM/radius and MOON_PA_DE421, not IAU_MOON or a
DE440 PA frame. Evaluating gravity in body axes then rotating acceleration
back to ICRF does not introduce rotating-frame fictitious forces.
"""
import ctypes
import hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, RotationSpline


class LunarHarmonics:
    def __init__(self, cosine, sine, gm, radius_m, *, frame, source):
        c,s=np.asarray(cosine,dtype=float),np.asarray(sine,dtype=float)
        if c.ndim!=2 or c.shape[0]!=c.shape[1] or s.shape!=c.shape or not np.all(np.isfinite([gm,radius_m])) or gm<=0 or radius_m<=0:
            raise ValueError("Square coefficients and positive finite GM/radius required")
        if not np.all(np.isfinite(c)) or not np.all(np.isfinite(s)) or not frame or not source:
            raise ValueError("Finite coefficients and explicit frame/source required")
        self.degree=c.shape[0]-1
        if not 0<=self.degree<=360 or c[0,0]!=1 or s[0,0]!=0:
            raise ValueError("Degree 0..360 and C00=1, S00=0 required")
        if np.any(np.triu(c,1)) or np.any(np.triu(s,1)) or np.any(s[:,0]):
            raise ValueError("Invalid order>degree coefficients or zonal sine coefficients")
        self.cosine=c.copy(); self.sine=s.copy()
        self.cosine.setflags(write=False); self.sine.setflags(write=False)
        self.gm=float(gm); self.radius_m=float(radius_m); self.frame=frame; self.source=source
        indices=np.tril_indices(self.degree+1)
        self._c=np.ascontiguousarray(c[indices]); self._s=np.ascontiguousarray(s[indices])
        self._native=None

    @classmethod
    def from_shadr(cls, path, degree=120, *, frame, source=None):
        """Read normalized PDS SHADR. Frame must come from the companion label.

        No rescaling to unrelated reference radii/GMs, no missing coefficient
        zero-filling, no double addition of J2. C00 is implicit in this format.
        """
        path=Path(path)
        if not isinstance(degree,(int,np.integer)) or not 0<=degree<=360:
            raise ValueError("Requested degree must be integer in 0..360")
        with path.open() as stream:
            h=[float(x.replace('D','E').replace('d','e')) for x in next(stream).strip().split(',')]
            if len(h)!=8 or h[5]!=1 or h[6]!=0 or h[7]!=0 or degree>min(h[3:5]):
                raise ValueError("Unsupported normalization, origin or degree in SHADR header")
            c=np.zeros((degree+1,degree+1)); s=c.copy(); seen=np.zeros_like(c,dtype=bool)
            c[0,0]=1; seen[0,0]=True
            for line in stream:
                values=line.strip().split(',')
                if len(values)!=6: raise ValueError("Malformed SHADR coefficient record")
                n,m=int(values[0]),int(values[1])
                if n>degree: break
                if not 0<=m<=n or n<0 or seen[n,m]: raise ValueError("Duplicate/invalid SHADR coefficient")
                c[n,m],s[n,m]=[float(x.replace('D','E').replace('d','e')) for x in values[2:4]]
                seen[n,m]=True
            if not np.all(seen[np.tril_indices(degree+1)]):
                raise ValueError("Truncated SHADR: requested coefficients are missing")
        result=cls(c,s,h[1]*1e9,h[0]*1e3,frame=frame,source=source or str(path.resolve()))
        result.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    def use_native(self, library_path):
        """Explicit host-compiled library, never builds or downloads implicitly."""
        self._library=ctypes.CDLL(str(library_path))
        array=np.ctypeslib.ndpointer(dtype=np.float64,ndim=1,flags="C_CONTIGUOUS")
        f=self._library.lunar_harmonics_acceleration
        f.argtypes=[array,ctypes.c_int,ctypes.c_double,ctypes.c_double,array,array,array,ctypes.POINTER(ctypes.c_double)]
        f.restype=ctypes.c_int
        self._native=f
        return self

    def evaluate(self, body_position):
        r=np.ascontiguousarray(body_position,dtype=float)
        if r.shape!=(3,) or not np.all(np.isfinite(r)) or np.linalg.norm(r)==0:
            raise ValueError("Finite nonzero body position required")
        if self._native is None:
            return self.evaluate_python(r)
        out=np.empty(3); potential=ctypes.c_double()
        if self._native(r,self.degree,self.gm,self.radius_m,self._c,self._s,out,ctypes.byref(potential)):
            raise ValueError("Native harmonic evaluation failed")
        return out,potential.value

    def evaluate_python(self, body_position):
        """Cartesian solid harmonics with analytic first derivatives; slow reference."""
        r=np.asarray(body_position,dtype=float)
        if r.shape!=(3,) or not np.all(np.isfinite(r)) or np.linalg.norm(r)==0:
            raise ValueError("Finite nonzero position required")
        rn=np.linalg.norm(r); r2=rn*rn; R=self.radius_m
        def mul(a,b): return np.r_[a[0]*b[0],a[1:]*b[0]+a[0]*b[1:]]
        xyz=[np.r_[R*r[i]/r2,R*np.eye(3)[i]/r2-2*R*r[i]*r/r2**2] for i in range(3)]
        X,Y,Z=xyz; Q=np.r_[R**2/r2,-2*R**2*r/r2**2]
        diag=np.r_[R/rn,-R*r/rn**3]; diag_s=np.zeros(4); total=np.zeros(4)
        for m in range(self.degree+1):
            if m:
                factor=np.sqrt(3 if m==1 else (2*m+1)/(2*m))
                diag,diag_s=factor*(mul(X,diag)-mul(Y,diag_s)),factor*(mul(Y,diag)+mul(X,diag_s))
            v2,w2=np.zeros(4),np.zeros(4); v1,w1=diag,diag_s
            for n in range(m,self.degree+1):
                v,w=v1,w1
                if n>m:
                    a=np.sqrt((4*n*n-1)/(n*n-m*m))
                    b=0 if n==m+1 else np.sqrt((2*n+1)*((n-1)**2-m*m)/((2*n-3)*(n*n-m*m)))
                    v,w=a*mul(Z,v1)-b*mul(Q,v2),a*mul(Z,w1)-b*mul(Q,w2)
                    v2,w2,v1,w1=v1,w1,v,w
                total+=self.cosine[n,m]*v+self.sine[n,m]*w
        total*=self.gm/R
        return total[1:],float(total[0])


class LunarOrientation:
    """Preload explicit SPICE body-to-ICRF rotations, then release added kernels.

    SPICE's process-global kernel pool is used during construction only. Construct
    this adapter serially. Interpolated evaluations are independent of SPICE.
    """
    def __init__(self,epoch_jd_tdb,duration_s,kernel_paths,*,frame="MOON_PA_DE421",step_s=600.0):
        if not np.all(np.isfinite([epoch_jd_tdb,duration_s,step_s])) or duration_s<=0 or step_s<=0:
            raise ValueError("Finite epoch and positive duration/step required")
        try: import spiceypy as spice
        except ImportError as exc: raise ImportError("Lunar orientation requires the navigation extra (spiceypy)") from exc
        self.frame=frame; self.duration_s=duration_s
        self.kernel_hashes={str(Path(p).resolve()):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in kernel_paths}
        loaded={str(Path(spice.kdata(i,'ALL')[0]).resolve()) for i in range(spice.ktotal('ALL'))}
        added=[]
        grid=np.linspace(0,duration_s,max(2,int(np.ceil(duration_s/step_s))+1))
        try:
            for path in self.kernel_hashes:
                if path not in loaded: spice.furnsh(path); added.append(path)
            epoch=(epoch_jd_tdb-2451545.0)*86400.0
            matrices=np.array([spice.pxform(frame,'J2000',epoch+t) for t in grid])
        finally:
            for path in reversed(added): spice.unload(path)
        self._rotations=RotationSpline(grid,Rotation.from_matrix(matrices))

    def __call__(self,time_s):
        if not np.isfinite(time_s) or not 0<=time_s<=self.duration_s:
            raise ValueError("Lunar orientation epoch outside preload coverage")
        return self._rotations(time_s).as_matrix()


class InertialLunarGravity:
    def __init__(self,field,orientation):
        if field.frame!=orientation.frame: raise ValueError("Gravity and orientation frames do not match")
        self.field=field; self.orientation=orientation

    def __call__(self,r,time_s):
        body_to_inertial=self.orientation(time_s)
        acceleration,_=self.field.evaluate(body_to_inertial.T @ r)
        return body_to_inertial @ acceleration
