"""Goal-aware candidate selection; Gamma is an admissibility bound, not a score.

Inspired by joint progress/tracking/control objectives in EVA-Planner and by
separating hard safety from performance in CBF-QP. This is an adaptation, not a
reproduction or a global arrival guarantee. Safety kernels are unchanged.
"""
import numpy as np
from deployment_detour import Detour, polytope, support, vertices, transport, detour_hold_margin
from sparse_projection import SparseProjectionQP


class GoalAwareDetour(Detour):
    def __init__(self, config, full_config, **kwargs):
        super().__init__(config, **kwargs)
        self.full_config=full_config
        self.context=None; self.clock=0.; self.last_search=-np.inf; self.last_switch=-np.inf
        self.candidate_qp=None; self.selection_count=0; self.last_selected_score=np.nan

    def set_context(self,x,goal,obs,log,B,C,target,previous,nominal,plan,dt,certified):
        self.context=dict(x=np.asarray(x).copy(),goal=np.asarray(goal).copy(),obs=obs,log=log,
          B=B,C=C,target=target,previous=previous,nominal=nominal,plan=plan,dt=dt,certified=certified)

    def goal_evaluation(self, control):
        c=self.context;cfg=self.full_config;x=c['x'];goal=c['goal']
        n=cfg['mpc_horizon'];ds=cfg['mpc_prediction_dt'];tt=(np.arange(n)+1)*ds
        delta=goal-x[:3];distance=np.linalg.norm(delta);direction=delta/max(distance,1e-12)
        desired=cfg['cruise_speed']*np.tanh(distance/1.6)*direction
        U=np.tile(c['nominal'],(n,1)) if c['plan'] is None else np.asarray(c['plan']).copy()
        if U.shape!=(n,3):raise ValueError('The nominal MPC plan must have horizon by 3 entries')
        U[0]=control
        v=x[3:].copy();p=x[:3].copy();value=0.
        previous=c['previous']
        for k in range(n):
            p=p+ds*v+.5*ds*ds*U[k];v=v+ds*U[k]
            ref=x[:3]+min(distance,tt[k]*np.linalg.norm(desired))*direction
            value+=(cfg['mpc_position_weight']*np.sum((p-ref)**2)
              +cfg['mpc_velocity_weight']*np.sum((v-desired)**2)
              +cfg['mpc_input_weight']*np.sum(U[k]**2)
              +cfg['mpc_increment_weight']*np.sum((U[k]-previous)**2))
            previous=U[k]
        return dict(cost=float(value),goal_distance_T=float(np.linalg.norm(goal-p)),
                    predicted_progress_m=float(distance-np.linalg.norm(goal-p)))

    def _candidate(self,d,A,b,e,z,ao,q,D,margin):
        d=np.asarray(d,float);d=d-e*(e@d);length=np.linalg.norm(d)
        if length<1e-10:return None
        d=d/length;capacity,_=support(A,b,d)
        gamma=capacity-d@ao-(self.cfg['k_perp']-q/D)*(d@z)
        executable=gamma-margin
        held=self.j>=0 and d@self.d>1-1e-9
        threshold=self.cfg['gamma_min']*(1 if held else 2)
        if executable<threshold:return None
        chi=-d@z;strength=min(self.cfg['nu_req'],self.cfg['eta']*executable)
        lower=d@ao+(q/D-self.cfg['k_perp'])*chi+strength+margin
        c=self.context;BB=np.r_[c['B'],d[None,:]];CC=np.r_[c['C'],-lower]
        if self.candidate_qp is None or self.candidate_qp.n!=len(CC):
            self.candidate_qp=SparseProjectionQP(len(CC))
        u,ok,_=self.candidate_qp.solve(c['target'],BB,CC)
        if not ok or not np.isfinite(u).all() or np.min(BB@u+CC)<-1e-7:return None
        score=self.goal_evaluation(u)
        return dict(d=d,gamma=float(gamma),score=score['cost'],progress=score['predicted_progress_m'])

    def choose(self,A,b,e,z,ao,q,D):
        if self.context is None:raise RuntimeError('Goal-aware direction search needs prediction context')
        c=self.context;cfg=self.full_config
        j=self.j
        if j<0:
            # Parent has selected the closest blocking candidate; recover its
            # identity from the supplied LOS (no future obstacle data used).
            vectors=c['obs'][:,:3]-c['x'][:3]
            ee=vectors/np.linalg.norm(vectors,axis=1)[:,None]
            j=int(np.argmin(np.linalg.norm(ee-e,axis=1)+abs(np.linalg.norm(vectors,axis=1)-D)/max(D,1.)))
        margin=detour_hold_margin(c['log'][j],c['dt'],self.cfg['k_perp']) if c['certified'] else 0.
        axis=np.eye(3)[np.argmin(np.abs(e))];n1=np.cross(e,axis);n1/=np.linalg.norm(n1);n2=np.cross(e,n1)
        directions=[]
        if self.j>=0:directions.append(self.d.copy())
        directions.extend([c['goal']-c['x'][:3],c['nominal']-ao])
        directions.extend(np.cos(a)*n1+np.sin(a)*n2 for a in np.arange(cfg['mpc_horizon'])*2*np.pi/cfg['mpc_horizon'])
        evaluated=[];seen=[]
        def add(d):
            d=np.asarray(d);d=d-e*(e@d);norm=np.linalg.norm(d)
            if norm<1e-10:return
            d=d/norm
            if any(d@old>1-1e-9 for old in seen):return
            seen.append(d);item=self._candidate(d,A,b,e,z,ao,q,D,margin)
            if item is not None:evaluated.append(item)
        for d in directions:add(d)
        # A finite direction stencil is not globally exhaustive. If it misses
        # all feasible directions, construct candidates from polytope vertices
        # and rank those with the SAME goal objective, never by largest Gamma.
        if not evaluated:
            for u in vertices(A,b):
                g=u-ao;g=g-e*(e@g)-(self.cfg['k_perp']-q/D)*z
                add(g)
        self.selection_count+=1
        if not evaluated:return None,-np.inf
        best=min(evaluated,key=lambda x:x['score'])
        if self.j>=0:
            held=next((r for r in evaluated if r['d']@self.d>1-1e-9),None)
            if held is not None and held['score']<=best['score']+1e-8:best=held
        self.last_selected_score=best['score']
        return best['d'],best['gamma']

    def update(self,x,goal,obs,log,B,C,nominal,dt,**kwargs):
        self.clock+=dt;changed=False
        # Existing prediction interval and existing release dwell are reused as
        # evaluation cadence and switch spacing; no seed-specific constants.
        if (self.enabled and self.j>=0 and self.clock-self.last_search>=self.full_config['mpc_prediction_dt']-1e-12
              and self.clock-self.last_switch>=self.cfg['tau_off']-1e-12):
            j=self.j;r=obs[j,:3]-x[:3];D=np.linalg.norm(r);e=r/D;w=obs[j,3:6]-x[3:];q=e@w;z=w-q*e
            self.d=transport(self.d,self.e,e);self.e=e
            A,b=polytope(B,C);d,gamma=self.choose(A,b,e,z,obs[j,6:9],q,D)
            self.last_search=self.clock
            if d is not None and d@self.d<1-1e-9:
                self.d=d;self.segment+=1;self.last_switch=self.clock;changed=True
        result=super().update(x,goal,obs,log,B,C,nominal,dt,**kwargs)
        if result['event']==1:self.last_search=self.clock;self.last_switch=self.clock
        if changed and result['active']:result.update(event=1,segment=self.segment,reason='goal_cost_reselection')
        result['goal_candidate_search_count']=self.selection_count
        result['goal_direction_cost']=self.last_selected_score
        return result
