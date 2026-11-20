"""Executable detour: chi, Gamma, exact vertices and transport.

The original release rule requires an unobstructed goal segment, nondecreasing
primary clearance, and a nominal input admissible in the continuous safety set
throughout tau_off. Sampling tightening is separate from that nominal test.
All-obstacle hard safety constraints remain in the QP.
"""
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection,QhullError
from itertools import combinations

def polytope(B,C,limit=2.):
    A=np.r_[B,np.eye(3),-np.eye(3)];b=np.r_[-C,np.full(6,-limit)]
    norm=np.linalg.norm(A,axis=1);return A/norm[:,None],b/norm

def vertices(A,b):
    center=linprog([0.,0.,0.,-1.],A_ub=np.c_[-A,np.ones(len(A))],b_ub=-b,
                   bounds=[(None,None)]*4,method='highs')
    if not center.success:return np.empty((0,3))
    if center.x[3]>1e-8:
        try:return HalfspaceIntersection(np.c_[-A,b],center.x[:3]).intersections
        except QhullError:pass
    out=[]
    for ids in combinations(range(len(A)),3):
        m=A[list(ids)]
        if abs(np.linalg.det(m))<1e-10:continue
        u=np.linalg.solve(m,b[list(ids)])
        if np.min(A@u-b)>=-1e-8:out.append(u)
    return np.array(out).reshape(-1,3)

def support(A,b,d):
    r=linprog(-d,A_ub=-A,b_ub=-b,bounds=[(None,None)]*3,method='highs')
    return (float(d@r.x),r.x) if r.success else (-np.inf,None)

def transport(d,old_e,e):
    """Shortest rotation is exact parallel transport on the connecting great arc.
    In the integrator this is applied every 0.5 ms, not only after a QP update.
    """
    axis=np.cross(old_e,e);s=np.linalg.norm(axis);c=float(old_e@e)
    if s>1e-12:
        unit=axis/s;d=d*c+np.cross(unit,d)*s+unit*(unit@d)*(1-c)
    d=d-e*(d@e);length=np.linalg.norm(d)
    return d/max(length,1e-15)

def blocked(p,goal,obstacles,R):
    line=goal-p;den=max(float(line@line),1e-15)
    fraction=np.clip((obstacles[:,:3]-p)@line/den,0.,1.)
    closest=p+fraction[:,None]*line
    return np.linalg.norm(obstacles[:,:3]-closest,axis=1)<=R

def detour_hold_margin(log,dt,kp,ao=.8,jerk=1.1):
    D,q,Z=log[:3];ar=ao+np.sqrt(12.);W=np.sqrt(q*q+Z)+dt*ar;lo=D-dt*W
    if lo<=1e-9:return np.inf
    Q=min(W,abs(q)+dt*(W*W/lo+ar));qd=W*W/lo+ar
    L=(W/lo)*ar+jerk+(qd/lo+Q*Q/lo**2)*W+(kp+Q/lo)*(ar+Q/lo*W)
    return dt*L+2e-7+0.5  # deployment tracking-error reserve

class Detour:
    def __init__(self,config,enabled=True,fixed_direction=False,fixed_strength=False):
        self.cfg=config;self.enabled=enabled;self.j=-1;self.d=np.zeros(3);self.e=np.array([1.,0.,0.])
        self.off_time=0.;self.segment=0;self.fixed_direction=fixed_direction
        self._previous_off_condition=False
        self.fixed_strength=fixed_strength

    def choose(self,A,b,e,z,ao,q,D):
        verts=vertices(A,b)
        if len(verts)==0:return None,-np.inf
        g=verts-ao;g-=np.outer(g@e,e);g-=(self.cfg['k_perp']-q/D)*z
        lengths=np.linalg.norm(g,axis=1);best=float(lengths.max())
        if best<1e-12:return None,0.
        if self.fixed_direction:
            d=np.cross(np.array([0.,0.,1.]),e)
            if np.linalg.norm(d)<1e-8:d=np.cross(np.array([0.,1.,0.]),e)
            d/=np.linalg.norm(d);value,_=support(A,b,d)
            return d,value-d@ao-(self.cfg['k_perp']-q/D)*(d@z)
        if self.j>=0:
            capacity,_=support(A,b,self.d)
            capacity-=self.d@ao+(self.cfg['k_perp']-q/D)*(self.d@z)
            if capacity>=best-1e-8:return self.d.copy(),capacity
        ids=np.flatnonzero(lengths>=best-1e-9)
        order=np.lexsort((verts[ids,2],verts[ids,1],verts[ids,0]));r=ids[order[0]]
        return g[r]/lengths[r],best

    def force_release(self):
        """Release only the auxiliary row; physical and safety states are untouched."""
        if self.j>=0:self.segment+=1
        self.j=-1;self.off_time=0.;self._previous_off_condition=False

    def update(self,x,goal,obs,log,B,C,nominal,dt,certified=True,safe_witness=None,
               nominal_admissible=None,activation_allowed=True):
        empty=dict(active=False,j=-1,d=np.zeros(3),chi=0.,gamma=0.,strength=0.,hold_margin=0.,
                   lower=0.,segment=self.segment,event=0,capacity_lp_calls=0,reason='disabled',
                   off_elapsed=0.,primary_clear=False,primary_receding=False,goal_clear=False,nominal_admissible=False)
        if not np.isfinite(dt) or dt<=0:raise ValueError('Positive finite update interval required.')
        if not self.enabled:return empty
        A,b=polytope(B,C);cfg=self.cfg;kp=cfg['k_perp'];mask=blocked(x[:3],goal,obs,log[:,3]);event=0
        # The auxiliary rule must only operate inside an already nonempty K.
        # Execution supplies a base-QP witness; standalone callers get an LP check.
        if safe_witness is None:
            _,safe_witness=support(A,b,np.zeros(3))
        if safe_witness is None or not np.isfinite(safe_witness).all() or np.min(A@safe_witness-b)<-1e-7:
            empty['reason']='base_set_infeasible';return empty
        if self.j<0 and not activation_allowed:
            empty['reason']='conditional_gate_closed';return empty
        if self.j<0:
            ids=np.flatnonzero(mask&(log[:,6]>=0)&(log[:,6]<=cfg['s_on']))
            if len(ids)==0:
                empty['reason']='no_blocking_candidate';return empty
            j=int(ids[np.lexsort((ids,log[ids,6]))[0]])
        else:j=self.j
        r=obs[j,:3]-x[:3];D=np.linalg.norm(r);e=r/D;w=obs[j,3:6]-x[3:];q=e@w;z=w-q*e
        if self.j>=0:self.d=transport(self.d,self.e,e)
        # Evaluate the original rule before capacity/reselection. The execution
        # caller supplies nominal admissibility in raw K, not tightened K_dt.
        clear=not bool(mask[j]);goal_clear=not bool(np.any(mask))
        receding=bool(q-log[j,4]>=0)
        if nominal_admissible is None:
            nominal_admissible=bool(np.isfinite(nominal).all() and
                np.max(np.abs(nominal))<=2.+1e-8 and np.min(B@nominal+C)>=-1e-8)
        nominal_admissible=bool(nominal_admissible)
        empty.update(primary_clear=clear,primary_receding=receding,
                     goal_clear=goal_clear,nominal_admissible=nominal_admissible)
        if self.j>=0 and goal_clear and receding and nominal_admissible:
            # First observed true predicate starts at zero elapsed time. Count
            # elapsed intervals, rather than crediting an unobserved future dt.
            self.off_time=self.off_time+dt if self._previous_off_condition else 0.
            self._previous_off_condition=True
        else:
            self.off_time=0.;self._previous_off_condition=False
        if self.j>=0 and self.off_time>=cfg['tau_off']-1e-12:
            elapsed=self.off_time;self.j=-1;self.off_time=0.;self._previous_off_condition=False
            self.segment+=1
            empty.update(event=2,segment=self.segment,reason='goal_clear_primary_receding_nominal_safe',
                         off_elapsed=elapsed,primary_clear=clear,primary_receding=receding)
            return empty
        margin=detour_hold_margin(log[j],dt,kp) if certified else 0.
        gamma=-np.inf
        if self.j>=0:
            sup,_=support(A,b,self.d);gamma=sup-self.d@obs[j,6:9]-(kp-q/D)*(self.d@z)
        if gamma-margin<cfg['gamma_min']:
            d,g=self.choose(A,b,e,z,obs[j,6:9],q,D)
            if d is None or g-margin<2*cfg['gamma_min']:
                if self.j>=0:self.segment+=1
                self.j=-1;self.off_time=0.;self._previous_off_condition=False
                empty.update(event=3,segment=self.segment,reason='insufficient_executable_capacity')
                return empty
            self.d=d;gamma=g;self.j=j;self.segment+=1;event=1
        self.e=e;chi=-self.d@z
        executable=gamma-margin
        strength=min(cfg['nu_req'],cfg['eta']*executable)
        if self.fixed_strength:
            # Ablation: request the same positive constant, without scaling by
            # Gamma. An unexecutable auxiliary request is paused, never imposed
            # by dropping a hard safety constraint or mislabeled as empty K.
            if cfg['nu_req']>executable:
                self.j=-1;self.off_time=0.;self._previous_off_condition=False;self.segment+=1
                empty.update(event=3,segment=self.segment,reason='fixed_strength_unexecutable')
                return empty
            strength=cfg['nu_req']
        lower=self.d@obs[j,6:9]+(q/D-kp)*chi+strength+margin
        return dict(active=True,j=j,d=self.d.copy(),chi=float(chi),gamma=float(gamma),
                    executable_gamma=float(executable),strength=float(strength),hold_margin=float(margin),
                    lower=float(lower),segment=self.segment,event=event,capacity_lp_calls=1,
                    reason='started_or_reselected' if event==1 else 'held',off_elapsed=self.off_time,
                    primary_clear=clear,primary_receding=receding,
                    goal_clear=goal_clear,nominal_admissible=nominal_admissible)
