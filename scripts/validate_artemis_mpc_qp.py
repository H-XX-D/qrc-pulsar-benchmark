#!/usr/bin/env python3
"""Independent bounded least-squares check of representative orbital MPC QPs."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import lsq_linear
from artemis_mpc_campaign import configurations
from qrc_pulsar_bench.artemis_pid_benchmark import PIDExperiment
from qrc_pulsar_bench.predictive_mpc import PredictiveMPC
from qrc_pulsar_bench.fpga_mpc import scaled_qp,float_solve,FixedQP

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','base','kernels','physics','native'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();experiment=PIDExperiment(a.base/'data',a.kernels,a.physics,a.native,{})
    solvers={b:FixedQP(a.output/f'mpc{b}.so',b,b-8) for b in (24,32)};rows=[]
    error=np.array([300.,-200.,100.,.005,-.003,.002])
    for case in ('P1_sep','P2_sep'):
        _,model,plan,_,_=experiment.prepare(case)
        for name,settings in configurations().items():
            controller=PredictiveMPC(model,plan,settings)
            for t in (0.,86400.,172800.):
                matrix,target,cap=controller.build_problem(t,plan.at(t)[:6]+error,500,np.full(settings.horizon_steps,120.))
                h,g,beta=scaled_qp(matrix,target);warm=np.zeros(len(g))
                ref=lsq_linear(matrix,target,bounds=(-1,1),tol=1e-12,max_iter=1000)
                if not ref.success:raise RuntimeError('Independent optimizer did not converge')
                reference_cost=float(np.linalg.norm(matrix@ref.x-target)**2)
                for arithmetic in ('float64','24','32'):
                    status,x,res=(float_solve(h,g,warm,beta,settings.residual_tolerance) if arithmetic=='float64'
                                  else solvers[int(arithmetic)].solve(h,g,warm,beta,settings.residual_tolerance))
                    rows.append(dict(case=case,time_s=t,configuration=name,arithmetic=arithmetic,status=status,residual=res,
                        first_dv_error_m_s=float(np.linalg.norm(x[:3]-ref.x[:3])*cap),
                        objective_excess_relative=float((np.linalg.norm(matrix@x-target)**2-reference_cost)/max(reference_cost,1e-12))))
    report=dict(classification='Representative orbital QPs, independent SciPy bounded least-squares comparison; not new gain-selection data',results=rows)
    (a.output/'qp-independent-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    for arithmetic in ('float64','24','32'):
        group=[r for r in rows if r['arithmetic']==arithmetic]
        print(arithmetic,'faults',sum(r['status']!=0 for r in group),'max first DV discrepancy',max(r['first_dv_error_m_s'] for r in group),'max relative objective excess',max(r['objective_excess_relative'] for r in group),flush=True)

if __name__=='__main__':main()
