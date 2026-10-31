"""Initial-state-only ARTEMIS replay against definitive navigation products.

SPKs and V02 CDFs are reconstructed orbit products, not raw observations or
independent truth. No later spacecraft state enters the propagation force.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from scipy.interpolate import make_interp_spline
from .adaptive_control import MoonCartesianFlightModel
from .cislunar_forces import CislunarForceParameters, acceleration_components
from .cislunar_perturbations import BarycentricEnvironment, CislunarPerturbations, PerturbationOptions
from .lunar_gravity import LunarHarmonics, LunarOrientation, InertialLunarGravity


DAY = 86400.
CASES = tuple(dict(name=f'P{probe}_{label}', probe=probe, utc=date, split=split)
              for label,date,split in [('sep','2011-09-01T00:00:00','training'),
                                       ('oct','2011-10-01T00:00:00','evaluation'),
                                       ('feb','2011-02-01T00:00:00','diagnostic')]
              for probe in (1,2))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_model(epoch_jd_tdb, duration, physics_dir, ephemeris, gm_kernel, native_library,
               *, cram=.016, degree=60, max_step_s=300., rtol=2e-13):
    """Same sourced force components as the control campaign at a real epoch.

    Cr*A/m is explicit. The replay trains it only on September arcs, not on
    an earlier forecast-versus-predicted-ephemeris calibration.
    """
    env=BarycentricEnvironment(epoch_jd_tdb,duration,ephemeris,gm_kernel,grid_step_s=300.)
    grid=env.spline.x;positions=env.spline(grid)
    inner_curve=make_interp_spline(grid,positions[:,0]-positions[:,1],k=5)
    inner_curve.extrapolate=False
    def inner(t):
        if not 0<=t<=duration:raise ValueError('Outside environment coverage')
        return inner_curve(t),inner_curve(t,1),inner_curve(t,2)
    orientation=LunarOrientation(epoch_jd_tdb,duration,
        [physics_dir/'moon_pa_de421_1900-2050.bpc',physics_dir/'moon_080317.tf'])
    field=LunarHarmonics.from_shadr(physics_dir/'gggrx_0900c_sha.tab',degree,
                                   frame='MOON_PA_DE421').use_native(native_library.resolve())
    lunar=InertialLunarGravity(field,orientation)
    extra=CislunarPerturbations(env,orientation,
        PerturbationOptions(earth_j2=True,other_planets=True,first_pn=True))
    params=CislunarForceParameters(gm_moon=field.gm,moon_radius_m=field.radius_m,
                                  moon_j2=0,area_mass_m2_kg=cram,reflectivity=1.)
    def acceleration(r,v,t):
        bodies,_=env(t)
        terms=acceleration_components(r,bodies[1]-bodies[0],bodies[2]-bodies[0],(0,0,1),params)
        return lunar(r,t)+terms['earth_tide']+terms['sun_tide']+terms['srp']+extra(r,v,t)
    return MoonCartesianFlightModel(acceleration,inner,moon_radius_m=field.radius_m,
             max_step_s=max_step_s,rtol=rtol),dict(epoch_jd_tdb=epoch_jd_tdb,
             parameters=asdict(params),extra=asdict(extra.options),lunar_degree=degree)


def load_arc(case,data,kernels,*,days=3.):
    import spiceypy as sp
    import cdflib
    from astropy.time import Time
    import erfa
    # Explicit kernel order, no dependence on a previous caller's SPICE pool.
    sp.kclear()
    for name in ['naif0012.tls','de440s.bsp','gm_de440.tpc']:sp.furnsh(str(kernels/name))
    path=data/f'ARTEMIS_P{case["probe"]}_DEF_2009022_2012022.bsp'
    sp.furnsh(str(path));body=str(-650-case['probe'])
    epoch=sp.str2et(case['utc']);times=np.arange(0,days*DAY+1,600.)
    states=np.array([sp.spkezr(body,epoch+t,'J2000','NONE','301')[0] for t in times])*1000.
    earth=np.array([sp.spkezr(body,epoch+t,'J2000','NONE','399')[0] for t in times])*1000.
    # Inspect the actual SPK segment frames and orbit-solution boundaries.
    handle=sp.dafopr(str(path));sp.dafbfs(handle);segments=[]
    while sp.daffna():
        bounds,meta=sp.dafus(sp.dafgs(),2,6)
        if meta[0]==int(body) and bounds[1]>=epoch and bounds[0]<=epoch+days*DAY:
            segments.append(dict(start_et=float(bounds[0]),end_et=float(bounds[1]),
                                 body=int(meta[0]),center=int(meta[1]),frame=int(meta[2]),type=int(meta[3])))
    sp.dafcls(handle)
    jumps=[]
    boundaries=sorted({v for s in segments for v in (s['start_et'],s['end_et']) if epoch<v<epoch+days*DAY})
    for b in boundaries:
        minus=np.array(sp.spkezr(body,b-.001,'J2000','NONE','399')[0])*1000
        plus=np.array(sp.spkezr(body,b+.001,'J2000','NONE','399')[0])*1000
        jumps.append(dict(time_s=b-epoch,position_jump_m=float(np.linalg.norm(plus[:3]-minus[:3]-.002*minus[3:])),
                          velocity_step_m_s=float(np.linalg.norm(plus[3:]-minus[3:]))))
    prefix='thb' if case['probe']==1 else 'thc'
    month=case['utc'][:7].replace('-','');cross=[];flags=[];files=[]
    for file in sorted(data.glob(f'{prefix}_l1_state_{month}*_v02.cdf')):
        cdf=cdflib.CDF(file);unix=cdf.varget(prefix+'_state_time')
        selected=(unix>=Time(case['utc'],scale='utc').unix)&(unix<=Time(case['utc'],scale='utc').unix+days*DAY)
        if not np.any(selected):continue
        indices=np.where(selected)[0][::10];t=Time(unix[indices],format='unix')
        rotation=erfa.pnm80(t.tt.jd1,t.tt.jd2)
        pos=np.einsum('nji,nj->ni',rotation,cdf.varget(prefix+'_pos')[indices])*1000
        et=(t.tdb.jd-2451545.)*DAY
        definitive=np.array([sp.spkezr(body,float(e),'J2000','NONE','399')[0][:3] for e in et])*1000
        cross.extend(np.linalg.norm(pos-definitive,axis=1).tolist())
        maneuver=np.asarray(cdf.varget(prefix+'_man'))[selected]
        flags.extend(np.unique(maneuver).tolist());files.append(dict(path=str(file),sha256=sha(file)))
    epoch_jd=2451545.+epoch/DAY
    meta=dict(case=case,classification='definitive orbit-product replay; reconstructed initial state may depend on later tracking',
         frame='Moon-relative J2000, geometric states, SI; elapsed TDB seconds',epoch_jd_tdb=epoch_jd,
         spk=dict(path=str(path),sha256=sha(path)),segments=segments,boundary_jumps=jumps,
         cdf_crosscheck=dict(version='V02 definitive',files=files,unique_maneuver_flags=sorted(set(flags)),
          position_rms_difference_m=None if not cross else float(np.sqrt(np.mean(np.square(cross)))),
          position_max_difference_m=None if not cross else float(max(cross)),
          note='Crosscheck only; orbit products may share OD inputs; flags and jumps are diagnostics, not a maneuver certificate.'))
    return times,states,meta


def replay(case,data,kernels,physics,native,cram,*,days=3.):
    times,observed,metadata=load_arc(case,data,kernels,days=days)
    model,model_inputs=make_model(metadata['epoch_jd_tdb'],days*DAY,physics,kernels/'de440s.bsp',
                                kernels/'gm_de440.tpc',native,cram=cram)
    start=time.process_time();sol=model.propagate(observed[0],0,days*DAY,500.)
    if not sol.completed:raise RuntimeError('Replay collided with a body')
    predicted=np.array([sol.at(t)[:6] for t in times]);errors=np.linalg.norm(predicted[:,:3]-observed[:,:3],axis=1)
    row=dict(metadata=metadata,model=model_inputs,cram=cram,cpu_s=time.process_time()-start,
             rms_m=float(np.sqrt(np.mean(errors[1:]**2))),max_m=float(max(errors)),
             endpoint_error_m={str(day):float(errors[np.argmin(abs(times-day*DAY))]) for day in (1,2,3)},
             initial_state_only=True,future_spacecraft_samples_in_force=False)
    return row,dict(time_s=times,reference=observed,prediction=predicted,error_m=errors)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--kernels',type=Path,required=True)
    p.add_argument('--physics',type=Path,required=True);p.add_argument('--native',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=[c['name'] for c in CASES],required=True)
    p.add_argument('--cram',type=float,default=.016);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    row,arrays=replay(next(c for c in CASES if c['name']==a.case),a.data,a.kernels,a.physics,a.native,a.cram)
    name=f'{a.case}-cram{a.cram:.3f}'
    (a.output/(name+'.json')).write_text(json.dumps(row,indent=2,allow_nan=False)+'\n')
    np.savez_compressed(a.output/(name+'.npz'),**arrays)
    print(a.case,a.cram,row['endpoint_error_m'],flush=True)


if __name__=='__main__':main()
