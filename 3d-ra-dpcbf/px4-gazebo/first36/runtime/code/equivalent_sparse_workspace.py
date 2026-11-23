"""Reuse allocations only; same CSC entries, solver construction and tolerances."""
import numpy as np
import scipy.sparse as sp
import clarabel
ENABLED=True
_installed=False
_work={}

def install():
 global _installed
 if _installed:return
 import sparse_projection as module
 original=module.clarabel_projection
 def projection(target,B,C):
  if not ENABLED:return original(target,B,C)
  n=len(C)
  if n not in _work:
   dense=np.empty((n+6,3));dense[n:n+3]=np.eye(3);dense[n+3:]=-np.eye(3)
   rhs=np.empty(n+6);rhs[n:]=2.
   settings=clarabel.DefaultSettings();settings.verbose=False;settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-11;settings.max_iter=400
   _work[n]=(dense,rhs,sp.eye(3,format='csc'),settings,[clarabel.NonnegativeConeT(n+6)])
  dense,rhs,P,settings,cones=_work[n];dense[:n]=-B;rhs[:n]=C
  A=sp.csc_matrix(dense)
  answer=clarabel.DefaultSolver(P,-np.asarray(target),A,rhs,cones,settings).solve()
  return np.asarray(answer.x) if str(answer.status) in ('Solved','AlmostSolved') else None
 module.clarabel_projection=projection;_installed=True
