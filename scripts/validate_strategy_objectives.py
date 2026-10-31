#!/usr/bin/env python3
"""Check sparse IRLS against an independent smooth sum-of-norms optimizer."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from qrc_pulsar_bench.artemis_pid_benchmark import PIDExperiment
from qrc_pulsar_bench.mpc_strategies import StrategyMPC,strategies
from qrc_pulsar_bench.predictive_mpc import PredictiveMPC
from qrc_pulsar_bench.fpga_mpc import scaled_qp,float_solve

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('output','base','kernels','physics','native'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();experiment=PIDExperiment(a.base/'data',a.kernels,a.physics,a.native,{})
    rows=[]
    for case in ('P1_sep','P2_sep'):
        _,model,plan,_,_=experiment.prepare(case)
        for name in ('sparse_01','sparse_05'):
            spec=strategies()[name];c=StrategyMPC(model,plan,spec)
            for amplitude in (20.,300.):
                state=plan.at(86400.)[:6]+np.array([amplitude,-amplitude/2,amplitude/3,.001,0,-.001])
                matrix,target,cap=PredictiveMPC.build_problem(c,86400.,state,500,np.full(4,120.))
                n=matrix.shape[1];x=np.zeros(n)
                for iteration in range(6):
                    norm=np.sqrt(np.sum(x.reshape(-1,3)**2,axis=1)+spec.sparse_epsilon**2)
                    weighted=np.vstack([matrix,np.diag(np.repeat(np.sqrt(spec.sparse_weight/norm),3))])
                    h,g,beta=scaled_qp(weighted,np.r_[target,np.zeros(n)])
                    status,x,res=float_solve(h,g,x,beta)
                    if status:raise RuntimeError('Sparse subproblem did not converge')
                def objective(u):
                    error=matrix@u-target;norm=np.sqrt(np.sum(u.reshape(-1,3)**2,axis=1)+spec.sparse_epsilon**2)
                    return .5*error@error+spec.sparse_weight*sum(norm)
                def gradient(u):
                    norm=np.sqrt(np.sum(u.reshape(-1,3)**2,axis=1)+spec.sparse_epsilon**2)
                    return matrix.T@(matrix@u-target)+spec.sparse_weight*u/np.repeat(norm,3)
                ref=minimize(objective,x,jac=gradient,bounds=[(-1,1)]*n,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-10,'maxiter':10000,'maxls':50})
                if not ref.success:raise RuntimeError(ref.message)
                rows.append(dict(case=case,strategy=name,offset_m=amplitude,relative_objective_excess=float((objective(x)-objective(ref.x))/objective(ref.x)),
                    first_dv_difference_m_s=float(np.linalg.norm(x[:3]-ref.x[:3])*cap)))
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'sparse-objective-audit.json').write_text(json.dumps(dict(classification='Fixed-B training-geometry objective check, not new strategy-selection data',results=rows),indent=2)+'\n')
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
