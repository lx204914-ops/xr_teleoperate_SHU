"""One execution path: all hard safety constraints, then executable detour.

Returns an explicit failure if the original safe set is empty. It never repairs
infeasibility by dropping obstacle constraints, relaxing them or resetting R.
"""
from dataclasses import dataclass
import time
import numpy as np
from detour import Detour
from conditional_gate import ConditionalDetourGate
from sparse_projection import SparseProjectionQP
from ra_dpcbf.certificates import current_common_set


def project_qp(target,B,C,backend):
    """Injection seam retained for the upstream numerical-failure tests."""
    return backend.solve(target,B,C)

VARIANTS=('detour','conditional_detour','no_detour','fixed_direction','fixed_radius','fixed_large',
          'no_transverse','fixed_strength','omit_Rdot_diagnostic','nominal')


@dataclass
class FilterResult:
    control: object
    status: str
    base_witness: object
    detour: dict
    qp_iterations: int
    minimum_raw_residual: object
    minimum_execution_residual: object
    minimum_augmented_residual: object
    auxiliary_binding: bool
    auxiliary_changed_input: bool
    direction_seconds: float
    qp_seconds: float


class SafetyFilter:
    def __init__(self,config,variant='detour'):
        if variant not in VARIANTS:raise ValueError('Unknown variant: '+variant)
        self.config=config
        self.variant=variant
        self.mechanism=Detour(config['detour'],
            enabled=variant not in ('no_detour','nominal'),
            fixed_direction=variant=='fixed_direction',fixed_strength=variant=='fixed_strength')
        self.conditional_gate=(ConditionalDetourGate(config['conditional_detour'])
                               if variant=='conditional_detour' else None)
        self.base_qp=None;self.augmented_qp=None

    def prepare(self,n):
        if self.base_qp is None or self.base_qp.n!=n:self.base_qp=SparseProjectionQP(n)
        if self.augmented_qp is None or self.augmented_qp.n!=n+1:self.augmented_qp=SparseProjectionQP(n+1)

    def solve(self,x,goal,obs,log,B,C,nominal,previous,dt,margins=None,certified=True):
        B=np.asarray(B,dtype=float);C=np.asarray(C,dtype=float)
        for name,value,shape in [('robot',x,(6,)),('goal',goal,(3,)),
            ('nominal',nominal,(3,)),('previous',previous,(3,))]:
            if np.shape(value)!=shape or not np.isfinite(value).all():
                raise ValueError(f'{name} must have shape {shape} and finite values.')
        margin=np.zeros_like(C) if margins is None else np.asarray(margins,dtype=float)
        if B.shape!=(len(obs),3) or C.shape!=(len(obs),) or margin.shape!=C.shape:
            raise ValueError('One hard constraint and margin are required for each obstacle.')
        if not np.isfinite(B).all() or not np.isfinite(C).all() or np.any(margin<0):
            raise ValueError('Invalid safety coefficients or negative margins.')
        off=dict(active=False,j=-1,d=np.zeros(3),chi=0.,gamma=0.,strength=0.,hold_margin=0.,
            lower=0.,segment=self.mechanism.segment,event=0,capacity_lp_calls=0,
            reason='base_set_not_checked',off_elapsed=0.,primary_clear=False,primary_receding=False)
        def failure(status,base=None,det=off,iterations=0,ds=0.,qs=0.):
            return FilterResult(None,status,base,det,iterations,None,None,None,False,False,ds,qs)
        if certified and np.any(np.linalg.norm(np.asarray(obs)[:,6:9],axis=1)>
                                self.config['obstacle_acceleration_bound']+1e-10):
            return failure('obstacle_acceleration_bound_violation')
        if not np.isfinite(margin).all():return failure('invalid_sampling_bound')
        Cexec=C-margin
        self.prepare(len(B))
        smoothing=self.config['parameters'][6]
        target=(np.asarray(nominal)+smoothing*np.asarray(previous))/(1+smoothing)
        tick=time.perf_counter()
        base,ok,base_it=project_qp(target,B,Cexec,self.base_qp)
        base_seconds=time.perf_counter()-tick
        if not ok:
            # A projection solver's failure is not itself an infeasibility proof.
            tick=time.perf_counter()
            execution=current_common_set(B,C,self.config['acceleration_component_limits'],margins=margin)
            raw=current_common_set(B,C,self.config['acceleration_component_limits'])
            base_seconds+=time.perf_counter()-tick
            if execution['feasible'] is False:
                label='base_qp_infeasible' if raw['feasible'] is False else 'sampling_set_infeasible'
            elif execution['feasible'] is True:label='base_qp_numerical_failure'
            else:label='feasibility_numerical_uncertainty'
            return failure(label,iterations=base_it,qs=base_seconds)
        if not np.isfinite(base).all() or (len(B) and np.min(B@base+Cexec)<-1e-7):
            return failure('base_qp_numerical_failure',iterations=base_it,qs=base_seconds)
        gate_info={"allow_activation":True,"gate_reason":"original_always_available",
                   "safe_progress":False,"release_elapsed_s":0.0}
        if self.conditional_gate is not None:
            gate_info=self.conditional_gate.pre_update(x,goal,obs,log,B,Cexec,target,base,dt,
                                                        self.mechanism.j>=0)
        tick=time.perf_counter()
        det=self.mechanism.update(x,goal,obs,log,B,Cexec,nominal,dt,
                                  certified=certified,safe_witness=base,
                                  nominal_admissible=bool(np.max(np.abs(nominal))<=2.+1e-8 and
                                      (not len(B) or np.min(B@nominal+C)>=-1e-8)),
                                  activation_allowed=gate_info['allow_activation'])
        det.update(gate_info)
        direction_seconds=time.perf_counter()-tick
        control=base;it=base_it;qp_seconds=base_seconds;minimum_augmented=None
        if det['active']:
            BB=np.r_[B,det['d'][None,:]];CC=np.r_[Cexec,-det['lower']]
            tick=time.perf_counter();control,ok,it=project_qp(target,BB,CC,self.augmented_qp)
            qp_seconds+=time.perf_counter()-tick
            if not ok or not np.isfinite(control).all():
                # Gamma claims feasibility: an inconsistent solver result is a
                # numerical failure, not evidence that the safety set is empty.
                return failure('augmented_qp_numerical_failure',base,det,it,direction_seconds,qp_seconds)
            minimum_augmented=float(det['d']@control-det['lower'])
            if minimum_augmented<-1e-7:
                return failure('augmented_qp_numerical_failure',base,det,it,direction_seconds,qp_seconds)
            if self.conditional_gate is not None:
                release=self.conditional_gate.post_augmented(x,goal,base,control,det,dt)
                det.update(release)
                if release['release']:
                    self.mechanism.force_release()
                    control=base;it=base_it;minimum_augmented=None
                    det.update(active=False,j=-1,strength=0.,lower=0.,event=4,
                               segment=self.mechanism.segment,reason='yield_for_goal_recovery',
                               safe_progress=True)
        raw=float(np.min(B@control+C)) if len(B) else None
        execution=float(np.min(B@control+Cexec)) if len(B) else None
        if execution is not None and execution<-1e-7:
            return failure('execution_qp_numerical_failure',base,det,it,direction_seconds,qp_seconds)
        if np.max(np.abs(control))>2+1e-7:
            return failure('input_box_numerical_failure',base,det,it,direction_seconds,qp_seconds)
        return FilterResult(control,'ok',base,det,it,raw,execution,minimum_augmented,
            bool(det['active'] and minimum_augmented is not None and abs(minimum_augmented)<=1e-6),
            bool(det['active'] and np.linalg.norm(control-base)>1e-7),direction_seconds,qp_seconds)
