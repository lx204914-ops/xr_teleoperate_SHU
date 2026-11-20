"""Bounded recovery after candidate-dependent tightening fails.

Uses the previous command only as a QP objective. All current hard constraints,
20ms held-input intervals and the unchanged 0.5 tracking ball must pass.
This is a feasible-input recovery, not a claim of nominal-cost optimality.
"""
import time
import numpy as np
from refine_park_certificate83 import certified_lower
from sparse_projection import SparseProjectionQP

def recover(controller,x,obs,B,C,tracking,extraB,extraC):
    tic=time.perf_counter();BB=np.r_[B,extraB];margin=np.zeros(len(C))
    backend=getattr(controller,'_certificate_recovery_qp',None)
    if backend is None:
        backend=SparseProjectionQP(len(BB));controller._certificate_recovery_qp=backend
    record=dict(accepted=False,iterations=0,minimum_certificate=None,scope='same constraints and interval verifier; alternate feasibility-search objective')
    for k in range(5):
        CC=np.r_[C-tracking-margin,extraC]
        u,ok,_=backend.solve(controller.previous,BB,CC)
        record['iterations']=k+1
        if not ok:break
        lo=certified_lower(x,obs,controller.par,u,controller.config['control_dt'],.5)
        record['minimum_certificate']=float(np.min(lo))
        if np.isfinite(lo).all() and lo.min()>=2e-7 and np.min(BB@u+np.r_[C-tracking,extraC])>=-2e-7:
            record.update(accepted=True,elapsed_ms=(time.perf_counter()-tic)*1000)
            return u,record
        if not np.isfinite(lo).all():break
        bad=lo<2e-7
        margin[bad]=np.maximum(margin[bad],(np.maximum(0.,B@u+C-tracking-lo)+1e-4)[bad])
    record['elapsed_ms']=(time.perf_counter()-tic)*1000
    return None,record
