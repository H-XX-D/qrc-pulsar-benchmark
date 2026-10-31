"""Matched-error benchmark of the existing spatial shape, Jacobi and Cartesian ODEs.

All three evolve the SAME isolated Newtonian Earth-Moon-test-particle system.
This isolates coordinate implementation costs; it is not a full-force flight
comparison. No later observed spacecraft states enter the benchmark or tuning.
Accuracy is measured against a refined independent Cartesian numerical control.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
from scipy.integrate import solve_ivp
from .capstone_replay import MultiDatasetFlightReplay
from .dynamics import RelationalState,RelationalThreeBodyIntegrator,CartesianThreeBodyIntegrator
from .shape_dynamics import integrate_spatial_shape
from .constants import G
from .cislunar_forces import GM_EARTH,GM_MOON


def benchmark(dataset,ephemeris,repeats=3):
    replay=MultiDatasetFlightReplay(dataset,ephemeris=str(ephemeris))
    earth=replay.environment._splines[0](0)
    earth_v=replay.environment._splines[0](0,1)
    f=GM_MOON/(GM_EARTH+GM_MOON)
    r1=-earth;v1=-earth_v
    r2=replay.initial_state[:3]+(1-f)*r1
    v2=replay.initial_state[3:]+(1-f)*v1
    masses=np.array([GM_EARTH/G,GM_MOON/G,1.])
    state=RelationalState(*masses,r1,v1,r2,v2)
    end=6*3600 if dataset=='lro_jan2023' else 3*86400
    times=np.linspace(0,end,1001)
    initial_jac=np.r_[r1,v1,r2,v2]
    initial_cart=np.concatenate(state.to_barycentric_cartesian())
    def cart_output(sol):
        y=sol.sol(times)
        return np.c_[(y[12:15]-y[6:9]).T,(y[15:18]-y[9:12]).T]
    def jac_output(y):
        return np.c_[(y[6:9]-(1-f)*y[:3]).T,(y[9:12]-(1-f)*y[3:6]).T]
    cart_rhs=lambda t,y:CartesianThreeBodyIntegrator._cartesian_derivatives(t,y,*masses)
    jac_rhs=lambda t,y:RelationalThreeBodyIntegrator._jacobi_derivatives(t,y,*masses)
    reference=[]
    for step in (120.,60.):
        sol=solve_ivp(cart_rhs,(0,end),initial_cart,method='DOP853',rtol=2e-14,
            atol=np.tile([1e-8]*3+[1e-11]*3,3),max_step=step,dense_output=True)
        if not sol.success: raise RuntimeError(sol.message)
        reference.append(cart_output(sol))
    control=reference[-1]
    reference_change=float(np.max(np.linalg.norm(reference[1][:,:3]-reference[0][:,:3],axis=1)))
    rows=[]
    # Alternate model order between repeats; include reconstruction of the same
    # 1001 output states in both wall and process CPU times.
    jobs=[(m,t) for t in (1e-8,1e-10,1e-12) for m in ('cartesian','jacobi','shape')]
    for repeat in range(repeats):
        order=jobs if repeat%2==0 else jobs[::-1]
        for model,rtol in order:
            start=time.perf_counter();cpu=time.process_time()
            try:
                if model=='shape':
                    atol=np.r_[np.full(7,rtol*1e-3),np.full(6,rtol*1e-8),rtol*1e-3]
                    result=integrate_spatial_shape(state,end,rtol=rtol,atol=atol)
                    sol=result.solution; predicted=jac_output(result.sol(times))
                    minimum_chart_v=float(np.min(sol.sol(times)[2]))
                else:
                    copies=3 if model=='cartesian' else 2
                    atol=np.tile([rtol*1e3]*3+[rtol]*3,copies)
                    sol=solve_ivp(cart_rhs if model=='cartesian' else jac_rhs,(0,end),
                        initial_cart if model=='cartesian' else initial_jac,method='DOP853',
                        rtol=rtol,atol=atol,dense_output=True)
                    if not sol.success: raise RuntimeError(sol.message)
                    predicted=cart_output(sol) if model=='cartesian' else jac_output(sol.sol(times))
                    minimum_chart_v=None
                row=dict(model=model,rtol=rtol,repeat=repeat,success=True,
                    wall_seconds=time.perf_counter()-start,cpu_seconds=time.process_time()-cpu,
                    nfev=sol.nfev,ode_dimension=sol.y.shape[0],accepted_steps=len(sol.t)-1,
                    max_position_error_m=float(np.max(np.linalg.norm(predicted[:,:3]-control[:,:3],axis=1))),
                    max_velocity_error_m_s=float(np.max(np.linalg.norm(predicted[:,3:]-control[:,3:],axis=1))),
                    minimum_chart_v=minimum_chart_v)
            except (ValueError,RuntimeError) as exc:
                row=dict(model=model,rtol=rtol,repeat=repeat,success=False,error=str(exc))
            rows.append(row)
            print(dataset,model,rtol,'error_m',row.get('max_position_error_m'), 'cpu_s',row.get('cpu_seconds'),flush=True)
    aggregates=[]
    for model,rtol in jobs:
        trials=[r for r in rows if r['model']==model and r['rtol']==rtol]
        if not all(r['success'] for r in trials): continue
        aggregates.append(dict(model=model,rtol=rtol,
            max_position_error_m=max(r['max_position_error_m'] for r in trials),
            median_wall_seconds=float(np.median([r['wall_seconds'] for r in trials])),
            median_cpu_seconds=float(np.median([r['cpu_seconds'] for r in trials])),
            min_cpu_seconds=min(r['cpu_seconds'] for r in trials),max_cpu_seconds=max(r['cpu_seconds'] for r in trials),
            nfev=trials[0]['nfev'],ode_dimension=trials[0]['ode_dimension']))
    matched={}
    for target in (10.,1.,.1):
        selected={}
        for model in ('cartesian','jacobi','shape'):
            candidates=[r for r in aggregates if r['model']==model and r['max_position_error_m']<=target]
            selected[model]=min(candidates,key=lambda r:r['median_cpu_seconds']) if candidates else None
        matched[str(target)]=selected
    return dict(dataset=dataset,duration_s=end,force_model='Same isolated Newtonian Earth-Moon-test-particle model in all solvers',
        reference='Conventional barycentric Cartesian DOP853; independently refined maximum step',
        reference_refinement_max_change_m=reference_change,output_epochs=len(times),
        caveat='Finite tolerance sweep measures these implementations on this CPU; not a theoretical optimum or full-force flight/FPGA benchmark',
        state_dimension_note='Spatial shape retains scale, quaternion, shape/rotation rates and an unused clock:14; Jacobi12; full Cartesian18. Removing unused clock gives13, still not a two-state system.',
        platform=platform.platform(),processor=platform.machine(),python=platform.python_version(),
        source_sha256={n:hashlib.sha256((Path(__file__).parent/n).read_bytes()).hexdigest() for n in
            ['dynamics.py','shape_dynamics.py','coordinate_benchmark.py']},
        trials=rows,aggregates=aggregates,matched_error_targets_m=matched)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=['lro_jan2023','capstone_feb2023'],required=True)
    p.add_argument('--ephemeris',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3)
    args=p.parse_args()
    if args.repeats<1: p.error('At least one repeat required')
    result=benchmark(args.dataset,args.ephemeris,args.repeats)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
