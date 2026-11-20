"""Isolated module switches for experiments; never used by the complete method."""
import numpy as np
from detour import Detour


class NoReselection(Detour):
    """Retain/transport the chosen side through capacity pauses until real release."""
    def choose(self,A,b,e,z,ao,q,D):
        if self.j>=0:return None,-np.inf
        return super().choose(A,b,e,z,ao,q,D)

    def update(self,x,goal,obs,log,B,C,nominal,dt,**kwargs):
        old_j=self.j;old_segment=self.segment;old_off=self.off_time
        old_previous=self._previous_off_condition
        result=super().update(x,goal,obs,log,B,C,nominal,dt,**kwargs)
        if old_j>=0 and result['reason']=='insufficient_executable_capacity':
            # Parent transported d to the current line of sight before testing
            # capacity. Keep it and its obstacle; only the auxiliary row pauses.
            self.j=old_j;self.segment=old_segment
            r=obs[old_j,:3]-x[:3];self.e=r/np.linalg.norm(r)
            predicate=bool(result['goal_clear'] and result['primary_receding'] and result['nominal_admissible'])
            self.off_time=old_off+dt if predicate and old_previous else 0.
            self._previous_off_condition=predicate
            result.update(segment=self.segment,off_elapsed=self.off_time,reason='held_direction_capacity_pause')
        return result
