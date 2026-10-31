"""Sourced force ablations and uncertainty scenarios, without residual fitting.

Run per dataset to save full forward states and a portable numerical receipt.
Tide conventions are sensitivity scenarios; their suitability for GRGM900C is
unresolved. All later spacecraft samples are retained and scored after solving.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .capstone_replay import MultiDatasetFlightReplay
from .lunar_gravity import LunarHarmonics,LunarOrientation,InertialLunarGravity
from .cislunar_perturbations import BarycentricEnvironment,CislunarPerturbations,PerturbationOptions
from .cislunar_refinement import physics_inputs,independent_event_windows,GRAIL


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',choices=list(MultiDatasetFlightReplay.DATASET_FILES),required=True)
    p.add_argument('--physics-dir',type=Path,required=True)
    p.add_argument('--ephemeris',type=Path,required=True)
    p.add_argument('--gm-kernel',type=Path,required=True)
    p.add_argument('--native-library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--initial-sensitivity',action='store_true')
    args=p.parse_args(); args.output.mkdir(parents=True,exist_ok=True)
    receipts=physics_inputs(args.physics_dir)
    replay=MultiDatasetFlightReplay(args.dataset,ephemeris=str(args.ephemeris),
        maneuver_evidence=independent_event_windows().get(args.dataset,()))
    orientation=LunarOrientation(replay.times_jd[0],replay.total_duration_s,
        [args.physics_dir/'moon_pa_de421_1900-2050.bpc',args.physics_dir/'moon_080317.tf'])
    field=LunarHarmonics.from_shadr(args.physics_dir/'gggrx_0900c_sha.tab',120,
        frame='MOON_PA_DE421',source=GRAIL+'gggrx_0900c_sha.tab').use_native(args.native_library.resolve())
    replay.lunar_gravity=InertialLunarGravity(field,orientation)
    env=BarycentricEnvironment(replay.times_jd[0],replay.total_duration_s,args.ephemeris,args.gm_kernel)
    duration=3.; step=30. if args.dataset.startswith('lro') else 300.
    models={
        'harmonics_only':PerturbationOptions(),
        'plus_earth_j2':PerturbationOptions(earth_j2=True),
        'plus_other_planets':PerturbationOptions(other_planets=True),
        'plus_first_pn':PerturbationOptions(first_pn=True),
        'all_no_tide':PerturbationOptions(earth_j2=True,other_planets=True,first_pn=True),
        'all_full_elastic_tide':PerturbationOptions(earth_j2=True,other_planets=True,first_pn=True,tide_convention='full_elastic'),
        'all_epoch_subtracted_tide':PerturbationOptions(earth_j2=True,other_planets=True,first_pn=True,tide_convention='epoch_subtracted'),
    }
    report=dict(dataset=args.dataset,days=duration,max_step_s=step,rtol=2e-12,
        protocol='First reconstructed spacecraft state only; preloaded planetary states; no reference-derived forces or fitted parameters',
        initial_state_caveat='Reconstructed initial state may itself use later OD observations; this is propagation replay, not a strictly operational blind forecast',
        physics_inputs=receipts,models={k:asdict(v) for k,v in models.items()},results=[],
        source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in
            ['cislunar_perturbations.py','capstone_replay.py','cislunar_error_budget.py']},
        gm_kernel_sha256=hashlib.sha256(args.gm_kernel.read_bytes()).hexdigest(),planetary_gm_m3_s2=env.gm.tolist())
    def save():
        (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    baseline=None; baseline_no_tide=None
    for name,options in models.items():
        extra=CislunarPerturbations(env,orientation,options)
        replay.additional_acceleration=extra if name!='harmonics_only' else None
        start=time.perf_counter(); result=replay.run_forward(duration,max_step_s=step,rtol=2e-12)
        row={k:v for k,v in result.items() if not isinstance(v,np.ndarray)}
        row.update(model=name,wall_seconds=time.perf_counter()-start)
        if baseline is None: baseline=result['states_si']
        if name=='all_no_tide': baseline_no_tide=result['states_si']
        change=np.linalg.norm(result['states_si'][:,:3]-baseline[:,:3],axis=1)
        row.update(max_change_from_harmonics_m=float(change.max()),rms_change_from_harmonics_m=float(np.sqrt(np.mean(change[1:]**2))))
        if options.tide_convention!='none':
            row['max_change_from_all_no_tide_m']=float(np.max(np.linalg.norm(result['states_si'][:,:3]-baseline_no_tide[:,:3],axis=1)))
        # Force amplitudes on the predicted path, never differentiated reference data.
        amplitudes={}
        for index in np.linspace(0,len(result['times_s'])-1,min(180,len(result['times_s'])),dtype=int):
            state=result['states_si'][index]
            for component,value in extra.components(state[:3],state[3:],result['times_s'][index]).items():
                amplitudes.setdefault(component,[]).append(float(np.linalg.norm(value)))
        row['sampled_acceleration_max_m_s2']={k:max(v) for k,v in amplitudes.items()}
        np.savez_compressed(args.output/(name+'.npz'),**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        report['results'].append(row);save()
        print(args.dataset,name,'RMS',row['rms_position_error_m'],'m; max change',row['max_change_from_harmonics_m'],flush=True)
    if args.initial_sensitivity:
        replay.additional_acceleration=CislunarPerturbations(env,orientation,models['all_no_tide'])
        times=result['times_s'];rows=[]
        for kind,magnitude,offset in [('position',100.,0),('velocity',.001,3)]:
            for axis in range(3):
                initial=replay.initial_state.copy();initial[offset+axis]+=magnitude
                predicted=replay.propagate(initial,times,max_step_s=step,rtol=2e-12)
                delta=np.linalg.norm(predicted[:,:3]-baseline_no_tide[:,:3],axis=1)
                rows.append(dict(kind=kind,axis=axis,magnitude_si=magnitude,
                    max_position_change_m=float(delta.max()),final_position_change_m=float(delta[-1])))
        report['initial_state_sensitivity']={'classification':'Illustrative one-axis perturbations; not a calibrated covariance or probability bound','cases':rows}
        save()


if __name__=='__main__': main()
