"""Equivalent numerical implementation; no change to the mathematical QPs.

Only rows proved redundant over the explicit input box are omitted from the
numerical workspace. Every returned answer is checked against ALL original rows.
"""
import inspect,textwrap
from itertools import combinations
import numpy as np
from numba import njit
from dynamic_study.kernel import radius_ref
ENABLED=True
QP_OPTIMIZATION=False
COUNTS=dict(qp_calls=0,rows_removed=0,target_feasible=0,support_calls=0,certified_support_hits=0)

def retained_rows(B,C):
    lo=np.full(3,-2.);hi=np.full(3,2.)
    axis=np.count_nonzero(B,axis=1)==1
    for k in np.flatnonzero(axis):
        j=int(np.flatnonzero(B[k])[0]);v=-C[k]/B[k,j]
        if B[k,j]>0:lo[j]=max(lo[j],v)
        else:hi[j]=min(hi[j],v)
    if np.any(lo>hi):return np.arange(len(C))
    minimum=C+np.sum(np.where(B>=0,B*lo,B*hi),axis=1)
    return np.flatnonzero(axis|(minimum<=1e-9))

@njit(cache=True)
def batch_refs(ref,x,obs,par):
    for i in range(len(obs)):
        r=obs[i,:3]-x[:3];D=np.linalg.norm(r)
        if D<=1e-9:raise ValueError('Undefined line of sight')
        q=r@(obs[i,3:6]-x[3:])/D
        ref[i],_=radius_ref(q,par[21]+par[22]+obs[i,9],par)

def install():
    from sparse_projection import SparseProjectionQP
    import px4_adapter_base as base
    from runtime_optimizer import SupportBackend
    if hasattr(SparseProjectionQP,'_exact_original_solve'):return
    original=SparseProjectionQP.solve;SparseProjectionQP._exact_original_solve=original
    original_init=SparseProjectionQP.__init__
    def initialize(self,n):
        original_init(self,n)
        if ENABLED:self.solver.update_settings(adaptive_rho_interval=25)
    SparseProjectionQP.__init__=initialize
    def solve(self,target,B,C,force_fallback=False):
        if not ENABLED or not QP_OPTIMIZATION:return original(self,target,B,C,force_fallback)
        target=np.asarray(target,float);B=np.asarray(B,float);C=np.asarray(C,float)
        COUNTS['qp_calls']+=1
        if not force_fallback and np.max(abs(target))<=2 and np.min(B@target+C)>=1e-10:
            COUNTS['target_feasible']+=1;return target.copy(),True,0
        ids=retained_rows(B,C);COUNTS['rows_removed']+=len(C)-len(ids)
        if len(ids)==len(C):return original(self,target,B,C,force_fallback)
        if not hasattr(self,'_exact_reduced'):self._exact_reduced={}
        if len(ids) not in self._exact_reduced:self._exact_reduced[len(ids)]=SparseProjectionQP(len(ids))
        answer,ok,it=original(self._exact_reduced[len(ids)],target,B[ids],C[ids],force_fallback)
        if ok and np.max(abs(answer))<=2+1e-7 and np.min(B@answer+C)>=-1e-7:return answer,ok,it
        return original(self,target,B,C,force_fallback)
    SparseProjectionQP.solve=solve
    original_support=SupportBackend.__call__
    def support(self,A,b,d):
        if not ENABLED:return original_support(self,A,b,d)
        COUNTS['support_calls']+=1
        # Exact LP primal/dual certificate. A feasible vertex and nonnegative
        # multipliers satisfying d=-M.T*lambda prove the same support optimum.
        for active in getattr(self,'_certified_bases',[]):
            if max(active)>=len(b):continue
            M=A[list(active)]
            try:
                witness=np.linalg.solve(M,b[list(active)])
                dual=np.linalg.solve(-M.T,d)
            except np.linalg.LinAlgError:continue
            if (np.min(dual)>=-1e-11 and np.max(abs(M.T@dual+d))<1e-10
                and np.min(A@witness-b)>=-1e-10):
                COUNTS['certified_support_hits']+=1
                return float(d@witness),witness
        ids=retained_rows(A,-b)
        value,witness=original_support(self,A[ids],b[ids],d)
        if witness is not None and np.min(A@witness-b)<-1e-9:value,witness=original_support(self,A,b,d)
        if witness is not None:
            tight=np.flatnonzero(abs(A@witness-b)<1e-7)
            bases=[]
            for active in combinations(tight,3):
                if abs(np.linalg.det(A[list(active)]))>1e-9:bases.append(tuple(map(int,active)))
                if len(bases)>=6:break
            self._certified_bases=bases
        return value,witness
    SupportBackend.__call__=support
    original_step=base.DeployedController.step
    source=textwrap.dedent(inspect.getsource(original_step))
    a=source.index('    for i in range(len(obs)):\n');b=source.index('    B,C,log=constraints',a)
    source=source[:a]+'    batch_refs(self.ref,x,obs,self.par)\n'+source[b:]
    namespace=dict(original_step.__globals__);namespace['batch_refs']=batch_refs
    exec(compile(source,original_step.__code__.co_filename,'exec'),namespace)
    compiled=namespace['step']
    def step(self,*args,**kw):return compiled(self,*args,**kw) if ENABLED else original_step(self,*args,**kw)
    base.DeployedController.step=step
    from px4_controller import Px4Controller
    original_advance=Px4Controller.advance_interval
    def advance(self,state,obstacles):
        if not ENABLED:return original_advance(self,state,obstacles)
        if self.last_result is None or self.last_result.control is None:raise RuntimeError('No executable current command')
        dt=self.config['control_dt']
        # Original substeps hold x, obstacles and Rref fixed: the analytic radius
        # flow has the semigroup property; LOS transport repeats the same e.
        base.boundary_flow(self.R,self.V,self.ref,dt,self.par[2])
        if self.last_result.detour['active']:
            mechanism=self.filter.mechanism;j=mechanism.j
            if j>=0:
                r=obstacles[j,:3]-state[:3];e=r/np.linalg.norm(r)
                mechanism.d=base.transport_step(mechanism.d,mechanism.e,e);mechanism.e=e.copy()
        self.time+=dt
    Px4Controller.advance_interval=advance
    # Warm the batched kernel before any flight starts.
    cfg=base.load_config(base.ROOT/'frozen_controller/method_config.json');p=np.asarray(cfg['parameters'])
    o=np.zeros((1,10));o[0,0]=10.;o[0,9]=.3;batch_refs(np.zeros(1),np.zeros(6),o,p)
