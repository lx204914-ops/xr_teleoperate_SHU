"""Warm strictly interior point for exact 3D support-polytope enumeration.

No obstacle rows, objectives or candidate directions are changed. If the old
interior point is no longer strictly interior, retain the original LP path.
"""
import numpy as np
from scipy.spatial import HalfspaceIntersection,QhullError
ENABLED=True
COUNTS=dict(warm_builds=0,warm_hits=0,rejected=0)

def install():
 from runtime_optimizer import SupportBackend,STATS
 old=SupportBackend.__call__
 def support(self,A,b,d):
  key=(A.shape,A.tobytes(),b.tobytes())
  if ENABLED and key!=self.key and getattr(self,'_warm_interior',None) is not None:
   center=self._warm_interior
   if np.min(A@center-b)>1e-7:
    try:
     points=HalfspaceIntersection(np.c_[-A,b],center).intersections
     if len(points) and np.isfinite(points).all() and np.min(A@points.T-b[:,None])>=-1e-10:
      self.key=key;self.queries=0;self.vertices=points;COUNTS['warm_builds']+=1
    except (QhullError,ValueError):COUNTS['rejected']+=1
  if ENABLED and key==self.key and self.vertices is not None and np.linalg.norm(d)>1e-15:
   STATS['queries']+=1;STATS['vertex_hits']+=1;COUNTS['warm_hits']+=1
   values=self.vertices@d;k=int(np.argmax(values));answer=float(values[k]),self.vertices[k].copy()
  else:answer=old(self,A,b,d)
  if self.vertices is not None:
   center=np.mean(self.vertices,axis=0)
   if np.min(A@center-b)>1e-7:self._warm_interior=center
  return answer
 SupportBackend.__call__=support
