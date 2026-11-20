"""Reuse candidate buffers; preserve original ADMM operations and tolerances."""
import sys,time,inspect,textwrap
import numpy as np
ENABLED=True
STATS=dict(candidate_calls=0,workspace_reuses=0,common_row_copies=0,reference_fallbacks=0,optimized_ms=0.)
_INSTALLED=False

def candidate_rows(self,c,d,lower):
 if not ENABLED:return np.r_[c['B'],d[None,:]],np.r_[c['C'],-lower]
 n=len(c['C'])
 if getattr(self,'_rows_context',None) is not c:
  if not hasattr(self,'_candidate_B') or self._candidate_B.shape!=(n+1,3):
   self._candidate_B=np.empty((n+1,3));self._candidate_C=np.empty(n+1)
  self._candidate_B[:n]=c['B'];self._candidate_C[:n]=c['C'];self._rows_context=c
  STATS['common_row_copies']+=1
 self._candidate_B[n]=d;self._candidate_C[n]=-lower
 return self._candidate_B,self._candidate_C

def install():
 global _INSTALLED
 if _INSTALLED:return
 from sparse_projection import SparseProjectionQP,clarabel_projection
 from goal_aware_detour import GoalAwareDetour
 original=SparseProjectionQP.solve
 def _candidate(self,target,B,C,force_fallback=False):
  return original(self,target,B,C,force_fallback)
 def solve(self,target,B,C,force_fallback=False):
  if sys._getframe(1).f_code.co_name!='_candidate':return original(self,target,B,C,force_fallback)
  if not ENABLED or force_fallback:return _candidate(self,target,B,C,force_fallback)
  tic=time.perf_counter();STATS['candidate_calls']+=1
  target=np.asarray(target,float);B=np.asarray(B,float);C=np.asarray(C,float)
  if B.shape!=(self.n,3) or C.shape!=(self.n,):raise ValueError('Hard-QP shape mismatch')
  if not hasattr(self,'_bounded_candidate'):
   self.solver.update_settings(max_iter=1000);self._bounded_candidate=True
  if not hasattr(self,'_candidate_lower'):
   self._candidate_lower=np.empty(self.n+3);self._candidate_lower[-3:]=-2.
   self._candidate_upper=np.full(self.n+3,np.inf);self._candidate_upper[-3:]=2.
  else:STATS['workspace_reuses']+=1
  matrix=self._matrix(B);self._candidate_lower[:self.n]=-C
  self.solver.update(q=-target,Ax=matrix.data,l=self._candidate_lower,u=self._candidate_upper)
  result=self.solver.solve(raise_error=False)
  good=result.info.status_val in (1,2) and result.x is not None
  if good:
   u=np.asarray(result.x);good=np.max(abs(u))<=2.+1e-7 and (self.n==0 or np.min(B@u+C)>=-1e-7)
  if not good:
   self.fallbacks+=1;STATS['reference_fallbacks']+=1;u=clarabel_projection(target,B,C)
   if u is None:return np.zeros(3),False,int(result.info.iter)
   good=np.max(abs(u))<=2.+1e-7 and (self.n==0 or np.min(B@u+C)>=-1e-7)
  elapsed=(time.perf_counter()-tic)*1000
  STATS['optimized_ms']+=elapsed
  from runtime_optimizer import STATS as runtime_stats
  runtime_stats['candidate_qp_ms']=runtime_stats.get('candidate_qp_ms',0.)+elapsed
  return u,bool(good),int(result.info.iter)
 source=textwrap.dedent(inspect.getsource(GoalAwareDetour._candidate))
 old="c=self.context;BB=np.r_[c['B'],d[None,:]];CC=np.r_[c['C'],-lower]"
 assert old in source
 scope=dict(GoalAwareDetour._candidate.__globals__,candidate_rows=candidate_rows)
 exec(compile(source.replace(old,'c=self.context;BB,CC=candidate_rows(self,c,d,lower)'),__file__,'exec'),scope)
 GoalAwareDetour._candidate=scope['_candidate'];SparseProjectionQP.solve=solve
 # Cache only immutable per-context scoring inputs, not a numerical optimum.
 from goal_eval_backend import evaluate
 old_goal=GoalAwareDetour.goal_evaluation
 def goal(self,control):
  if not ENABLED:return old_goal(self,control)
  c=self.context;cfg=self.full_config;n=cfg['mpc_horizon']
  if getattr(self,'_scoring_context',None) is not c:
   U=np.tile(c['nominal'],(n,1)) if c['plan'] is None else np.asarray(c['plan'])
   if U.shape!=(n,3):raise ValueError('The nominal MPC plan must have horizon by 3 entries')
   self._scoring_U=U
   self._scoring_weights=np.array([cfg['mpc_position_weight'],cfg['mpc_velocity_weight'],cfg['mpc_input_weight'],cfg['mpc_increment_weight']])
   self._scoring_context=c
  cost,end,progress=evaluate(c['x'],c['goal'],c['nominal'],self._scoring_U,c['previous'],np.asarray(control),cfg['mpc_prediction_dt'],cfg['cruise_speed'],self._scoring_weights)
  return dict(cost=float(cost),goal_distance_T=float(end),predicted_progress_m=float(progress))
 GoalAwareDetour.goal_evaluation=goal
 # Warm geometry previously recomputed the SAME vertex centroid for each of
 # the 14 directions. Reuse that exact np.mean result until vertices change.
 from runtime_optimizer import SupportBackend
 warm=SupportBackend.__call__;source=textwrap.dedent(inspect.getsource(warm))
 needle="if self.vertices is not None:\n  center=np.mean(self.vertices,axis=0)\n  if np.min(A@center-b)>1e-7:self._warm_interior=center"
 replacement="if self.vertices is not None and (not candidate_cache_enabled() or getattr(self,'_centroid_vertices',None) is not self.vertices):\n  center=np.mean(self.vertices,axis=0)\n  if np.min(A@center-b)>1e-7:self._warm_interior=center\n  self._centroid_vertices=self.vertices"
 assert needle in source
 ns=dict(warm.__globals__)
 ns.update({key:cell.cell_contents for key,cell in zip(warm.__code__.co_freevars,warm.__closure__ or ())})
 ns['candidate_cache_enabled']=lambda:ENABLED
 exec(compile(source.replace(needle,replacement),__file__,'exec'),ns)
 SupportBackend.__call__=ns['support'];_INSTALLED=True
