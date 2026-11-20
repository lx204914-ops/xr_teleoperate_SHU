"""Independent selector copy. All constraints, feasibility, transport and exits unchanged.

Progress replaces composite nominal tracking as the primary preference. An
already feasible direction is retained unless the *predicted* endpoint gain
exceeds one nominal cruise control tick. This is a local heuristic and not a
global navigation or future safety certificate. No future obstacle tape is used.
"""
import numpy as np
from goal_aware_detour import GoalAwareDetour
from detour import detour_hold_margin,vertices,support
from sparse_projection import SparseProjectionQP

class StableProgressDetour(GoalAwareDetour):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.decisions=[]
 def _candidate(self,d,A,b,e,z,ao,q,D,margin):
  if not self.fixed_strength:return super()._candidate(d,A,b,e,z,ao,q,D,margin)
  # Fixed-strength ablation must score exactly the auxiliary row it executes.
  d=np.asarray(d,float);d=d-e*(e@d);norm=np.linalg.norm(d)
  if norm<1e-10:return None
  d/=norm;capacity,_=support(A,b,d)
  gamma=capacity-d@ao-(self.cfg['k_perp']-q/D)*(d@z);executable=gamma-margin
  held=self.j>=0 and d@self.d>1-1e-9
  if executable<max(self.cfg['nu_req'],self.cfg['gamma_min']*(1 if held else 2)):return None
  chi=-d@z;lower=d@ao+(q/D-self.cfg['k_perp'])*chi+self.cfg['nu_req']+margin
  c=self.context;BB=np.r_[c['B'],d[None,:]];CC=np.r_[c['C'],-lower]
  if self.candidate_qp is None or self.candidate_qp.n!=len(CC):self.candidate_qp=SparseProjectionQP(len(CC))
  u,ok,_=self.candidate_qp.solve(c['target'],BB,CC)
  if not ok or not np.isfinite(u).all() or np.min(BB@u+CC)<-1e-7:return None
  score=self.goal_evaluation(u)
  return dict(d=d,gamma=float(gamma),score=score['cost'],progress=score['predicted_progress_m'])
 def choose(self,A,b,e,z,ao,q,D):
  if self.context is None:raise RuntimeError('Missing current-state prediction context')
  c=self.context;cfg=self.full_config;j=self.j
  if j<0:
   r=c['obs'][:,:3]-c['x'][:3];dd=np.linalg.norm(r,axis=1);ee=r/dd[:,None]
   j=int(np.argmin(np.linalg.norm(ee-e,axis=1)+abs(dd-D)/max(D,1.)))
  margin=detour_hold_margin(c['log'][j],c['dt'],self.cfg['k_perp']) if c['certified'] else 0.
  axis=np.eye(3)[np.argmin(abs(e))];n1=np.cross(e,axis);n1/=np.linalg.norm(n1);n2=np.cross(e,n1)
  candidates=([self.d.copy()] if self.j>=0 else [])+[c['goal']-c['x'][:3],c['nominal']-ao]
  candidates += [np.cos(a)*n1+np.sin(a)*n2 for a in np.arange(cfg['mpc_horizon'])*2*np.pi/cfg['mpc_horizon']]
  evaluated=[];seen=[]
  def add(v):
   v=np.asarray(v);v=v-e*(e@v);norm=np.linalg.norm(v)
   if norm<1e-10:return
   v=v/norm
   if any(v@w>1-1e-9 for w in seen):return
   seen.append(v);item=self._candidate(v,A,b,e,z,ao,q,D,margin)
   if item is not None:evaluated.append(item)
  for d in candidates:add(d)
  if not evaluated:
   for u in vertices(A,b):
    g=u-ao;g-=e*(e@g)+(self.cfg['k_perp']-q/D)*z;add(g)
  self.selection_count+=1
  if not evaluated:return None,-np.inf
  best=max(evaluated,key=lambda a:(a['progress'],-a['score']))
  held=next((a for a in evaluated if self.j>=0 and a['d']@self.d>1-1e-9),None)
  gain=None if held is None else float(best['progress']-held['progress'])
  threshold=float(cfg['cruise_speed']*c['dt'])
  tolerance=1e-8*max(1.,abs(best['progress']),abs(held['progress']) if held is not None else 0.)
  if held is not None and gain<=threshold+tolerance:best=held
  self.last_selected_score=float(best['score'])
  self.decisions.append(dict(time_s=float(self.clock),j=int(j),held_valid=held is not None,
    best_progress_gain_m=gain,switch_threshold_m=threshold,
    chosen_progress_m=float(best['progress']),chosen_cost=float(best['score']),
    chosen_direction=best['d'].tolist(),held_direction=self.d.tolist() if self.j>=0 else None,
    held_progress_m=None if held is None else float(held['progress']),
    candidates=len(evaluated),decision='initial_or_capacity' if held is None else ('hold' if best is held else 'progress_switch')))
  return best['d'],best['gamma']
