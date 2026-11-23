"""Equivalent numerical backends; no CBF, candidate, score, or lifecycle changes."""
import sys,warnings,time
from types import SimpleNamespace
import numpy as np
from scipy.optimize import linprog,OptimizeWarning
from scipy.spatial import HalfspaceIntersection,QhullError

_installed=False
STATS=dict(queries=0,direct_lps=0,vertex_lps=0,vertex_hits=0,fallbacks=0)
def serial_lp(*args,**kwargs):
    options=dict(kwargs.get('options') or {});options.update(threads=1,parallel=False);kwargs['options']=options
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',message='Unrecognized options detected:.*',category=OptimizeWarning)
        return linprog(*args,**kwargs)

class SupportBackend:
    def __init__(self,original):self.original=original;self.key=None;self.queries=0;self.vertices=None
    def __call__(self,A,b,d):
        STATS['queries']+=1
        key=(A.shape,A.tobytes(),b.tobytes())
        if key!=self.key:self.key=key;self.queries=0;self.vertices=None
        self.queries+=1
        # Single held-direction queries keep the original direct LP. Multiple
        # objectives over the SAME bounded polytope reuse its exact vertices.
        if self.queries==2:
            STATS['vertex_lps']+=1
            center=serial_lp([0.,0.,0.,-1.],A_ub=np.c_[-A,np.ones(len(A))],b_ub=-b,bounds=[(None,None)]*4,method='highs')
            if center.success and center.x[3]>1e-7:
                try:
                    vertices=HalfspaceIntersection(np.c_[-A,b],center.x[:3]).intersections
                    if len(vertices) and np.isfinite(vertices).all() and np.min(A@vertices.T-b[:,None])>=-1e-10:self.vertices=vertices
                except QhullError:pass
        if self.vertices is not None and np.linalg.norm(d)>1e-15:
            values=self.vertices@d;k=int(np.argmax(values));STATS['vertex_hits']+=1
            return float(values[k]),self.vertices[k].copy()
        STATS['direct_lps']+=1
        if self.queries>1:STATS['fallbacks']+=1
        return self.original(A,b,d)

def fast_matrix(self,B):
    if not hasattr(self,'_direct_matrix_buffer'):
        data=np.empty(3*(self.n+1));self._direct_matrix_buffer=SimpleNamespace(data=data,nnz=self.nnz)
    blocks=self._direct_matrix_buffer.data.reshape(3,self.n+1)
    blocks[:,:self.n]=B.T;blocks[:,self.n]=1.
    return self._direct_matrix_buffer

def install():
    global _installed
    if _installed:return
    import detour,goal_aware_detour,stable_progress
    from ra_dpcbf import certificates
    from sparse_projection import SparseProjectionQP
    # Keep original function code and replace only the LP numerical transport.
    detour.linprog=serial_lp;certificates.linprog=serial_lp
    backend=SupportBackend(detour.support)
    for module in (detour,goal_aware_detour,stable_progress):module.support=backend
    SparseProjectionQP._matrix=fast_matrix
    from goal_eval_backend import install as install_goal_backend
    install_goal_backend()
    from dense_exact_backend import install as install_dense
    install_dense()
    from warm_geometry_backend import install as install_warm
    install_warm()
    from kkt_warm_backend import install as install_kkt
    install_kkt()
    _installed=True

def stats():return dict(STATS)
