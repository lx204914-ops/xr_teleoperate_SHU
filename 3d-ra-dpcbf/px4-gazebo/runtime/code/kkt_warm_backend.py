"""Certified warm starts only: original full OSQP solve is always retained."""
import numpy as np
import sys
import time
from itertools import combinations
from numba import njit
ENABLED=True
COUNTS=dict(calls=0,certified_warm_starts=0,uncertified_fallbacks=0)

@njit(cache=True)
def certificate(target,B,C,sets):
 n=len(C);zero=np.zeros(n);good=True
 for i in range(n):
  if np.dot(B[i],target)+C[i]<-1e-12:good=False;break
 if good:return target.copy(),zero,True
 for row in sets:
  k=0
  for j in range(3):
   if row[j]>=0:k+=1
  if k==0:continue
  H=np.empty((k,3));rhs=np.empty(k);norm=np.empty(k)
  for j in range(k):
   norm[j]=np.linalg.norm(B[row[j]])
   if norm[j]<1e-15:norm[j]=1.
   H[j]=B[row[j]]/norm[j];rhs[j]=-C[row[j]]/norm[j]-np.dot(H[j],target)
  gram=H@H.T
  if abs(np.linalg.det(gram))<1e-11:continue
  lam=np.linalg.solve(gram,rhs)
  if np.min(lam)<-1e-11:continue
  u=target+H.T@lam
  feasible=True
  for i in range(n):
   if np.dot(B[i],u)+C[i]<-1e-9:feasible=False;break
  if not feasible:continue
  dual=np.zeros(n)
  for j in range(k):dual[row[j]]=lam[j]/norm[j]
  return u,dual,True
 return target.copy(),zero,False

def install():
 from sparse_projection import SparseProjectionQP
 old=SparseProjectionQP.solve
 def solve(self,target,B,C,force_fallback=False):
  # Candidate ranking is sensitive to tiny numerical tie changes. Preserve its
  # original solver path exactly; accelerate only base/reach/final projection.
  if not ENABLED or force_fallback:return old(self,target,B,C,force_fallback)
  if sys._getframe(1).f_code.co_name=='_candidate':
   # Keep the original ADMM path and tolerance; only bound long iterations.
   # Original solve() invokes its existing Clarabel fallback on nonconvergence.
   if not hasattr(self,'_bounded_candidate'):
    self.solver.update_settings(max_iter=1000);self._bounded_candidate=True
   tic=time.perf_counter();answer=old(self,target,B,C,force_fallback)
   from runtime_optimizer import STATS
   STATS['candidate_solver_ms']=STATS.get('candidate_solver_ms',0.)+(time.perf_counter()-tic)*1000
   return answer
  COUNTS['calls']+=1;start=time.perf_counter();target=np.asarray(target,float);B=np.asarray(B,float);C=np.asarray(C,float);n=len(C)
  if not hasattr(self,'_bounded_iterations'):
   # Same tolerance and original high-accuracy Clarabel fallback. Never execute
   # a max-iteration OSQP iterate; the original solve checks status and falls back.
   self.solver.update_settings(max_iter=1000);self._bounded_iterations=True
  A=np.r_[B,np.eye(3),-np.eye(3)];cc=np.r_[C,np.full(6,2.)]
  slack=(A@target+cc)/np.maximum(np.linalg.norm(A,axis=1),1e-15)
  pool=list(dict.fromkeys(list(getattr(self,'_kkt_previous',()))+list(np.argsort(slack)[:7])+[n-1]))
  pool=[i for i in pool if 0<=i<len(cc)][:5]
  bases=[]
  for k in (1,2,3):
   bases.extend(tuple(v)+(-1,)*(3-k) for v in combinations(pool,k))
  sets=np.asarray(bases,dtype=np.int64)
  u,dual,ok=certificate(target,A,cc,sets)
  if ok:
   # B u+C >=0 has OSQP lower-bound dual -lambda. The final six
   # certificate rows represent OSQP's existing two-sided +/-2 box.
   y=np.r_[-dual[:n],-dual[n:n+3]+dual[n+3:n+6]]
   self.solver.warm_start(x=u,y=y);COUNTS['certified_warm_starts']+=1
  else:COUNTS['uncertified_fallbacks']+=1
  warm_end=time.perf_counter();answer,success,iterations=old(self,target,B,C,force_fallback);solve_end=time.perf_counter()
  from runtime_optimizer import STATS
  STATS['kkt_certificate_ms']=STATS.get('kkt_certificate_ms',0.)+(warm_end-start)*1000
  STATS['projection_solver_ms']=STATS.get('projection_solver_ms',0.)+(solve_end-warm_end)*1000
  if success:
   tight=np.flatnonzero(abs(A@answer+cc)<1e-6);self._kkt_previous=tuple(map(int,tight[:6]))
  return answer,success,iterations
 SparseProjectionQP.solve=solve
 from sparse_projection import clarabel_projection
 probe=clarabel_projection(np.zeros(3),np.r_[np.eye(3),-np.eye(3)],np.ones(6))
 assert probe is not None and np.linalg.norm(probe)<1e-8
 # Compile the certificate before starting PX4, including the LAPACK path.
 A=np.r_[np.eye(3),-np.eye(3)];certificate(np.array([2.,0.,0.]),A,np.ones(6),np.array([[3,-1,-1]],dtype=np.int64))
