"""Only a controlled ablation; full/goal-aware controller source is untouched.

Same goal-aware first direction, parallel transport, strength, gate and exits.
Lock a direction through capacity pauses. Re-arm selection only after a normal
or gate-driven release, never by pausing and silently picking another side.
"""
from context47 import ROOT
from stable_progress import StableProgressDetour as GoalAwareDetour
from detour import Detour,transport,support
import numpy as np
from numba import njit
from scene_tapes import state_at
from evaluation_run import transport_step

@njit(cache=True)
def paused_direction_transport(p,v,u,tape,td,mode,t,dt,fine,j,d,e):
 # Update only the auxiliary memory along the same executed fine-step motion.
 # It neither changes physical integration nor supplies future data to a QP.
 p=p.copy();v=v.copy();d=d.copy();e=e.copy()
 for k in range(int(round(dt/fine))):
  p+=fine*v+.5*fine*fine*u;v+=fine*u
  obs=state_at(tape,t+(k+1)*fine,td,mode)
  r=obs[j,:3]-p;ne=r/np.linalg.norm(r)
  d=transport_step(d,e,ne);e=ne
 return d,e

class NoReselection(GoalAwareDetour):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs);self.lock_j=None;self.lock_d=None;self.lock_e=None
 def choose(self,A,b,e,z,ao,q,D):
  if self.lock_j is None:return GoalAwareDetour.choose(self,A,b,e,z,ao,q,D)
  d=self.d.copy()
  capacity,_=support(A,b,d)
  gamma=capacity-d@ao-(self.cfg['k_perp']-q/D)*(d@z)
  return d,float(gamma)
 def force_release(self):
  Detour.force_release(self);self.lock_j=None;self.lock_d=None;self.lock_e=None
 def update(self,x,goal,obs,log,B,C,nominal,dt,**kwargs):
  self.clock+=dt
  # Continue the same episode if its auxiliary row was temporarily paused.
  if self.lock_j is not None and self.j<0:
   if not kwargs.get('activation_allowed',True):
    e=obs[self.lock_j,:3]-x[:3];e=e/np.linalg.norm(e)
    self.lock_d=transport(self.lock_d,self.lock_e,e);self.lock_e=e
    return Detour.update(self,x,goal,obs,log,B,C,nominal,dt,**kwargs)
   self.j=self.lock_j;self.d=self.lock_d.copy();self.e=self.lock_e.copy()
  existing=self.j
  result=Detour.update(self,x,goal,obs,log,B,C,nominal,dt,**kwargs)
  if result['active']:
   self.lock_j=self.j;self.lock_d=self.d.copy();self.lock_e=self.e.copy()
  elif result['event']==2:
   self.lock_j=None;self.lock_d=None;self.lock_e=None
  elif self.lock_j is not None and existing>=0:
   # Parent already transported d before checking capacity; retain its NEW LOS.
   e=obs[self.lock_j,:3]-x[:3];e=e/np.linalg.norm(e)
   self.e=e;self.lock_d=self.d.copy();self.lock_e=e.copy()
  return result
