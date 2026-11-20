"""Computable versions of the latest manuscript's feasibility statements.

Continuous-time K and the sampling-tightened execution set are separate objects.
`not_certified` never means that the current QP is infeasible. No radius state is
reset by any check. Reported certificates are floating-point checks of explicit
inequalities, not interval-arithmetic/computer-assisted proofs.
"""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linprog
from .core import Boundary, evaluate, los, reference_radius


def _array(value, shape, name):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f'{name} must have shape {shape} and finite entries.')
    return out


@dataclass(frozen=True)
class InitialState:
    """The data required by the finite-interval proposition, with no future data."""
    relative_position: np.ndarray
    relative_velocity: np.ndarray
    base_radius: np.ndarray
    radius: np.ndarray
    rate: np.ndarray
    obstacle_acceleration_bound: np.ndarray

    def __post_init__(self):
        r = np.asarray(self.relative_position, dtype=float)
        if r.ndim != 2 or r.shape[1] != 3 or len(r) == 0 or not np.isfinite(r).all():
            raise ValueError('relative_position must be a nonempty finite (N,3) array.')
        n = len(r)
        object.__setattr__(self, 'relative_position', r.copy())
        object.__setattr__(self, 'relative_velocity', _array(self.relative_velocity, (n,3), 'relative_velocity').copy())
        for key in ('base_radius', 'radius', 'rate', 'obstacle_acceleration_bound'):
            object.__setattr__(self, key, _array(getattr(self,key), (n,), key).copy())
        if np.any(np.linalg.norm(r, axis=1) <= 0) or np.any(self.base_radius <= 0):
            raise ValueError('All center distances and base radii must be positive.')
        if np.any(self.obstacle_acceleration_bound < 0):
            raise ValueError('Acceleration bounds must be nonnegative.')


def initial_state(robot, obstacles, boundaries, cfg):
    robot = _array(robot, (6,), 'robot')
    obs = np.asarray(obstacles, dtype=float)
    if obs.ndim != 2 or obs.shape[1] != 10 or len(obs) != len(boundaries) or not np.isfinite(obs).all():
        raise ValueError('obstacles must be finite (N,10), with one boundary per obstacle.')
    return InitialState(obs[:,:3]-robot[:3], obs[:,3:6]-robot[3:],
        np.asarray([b.base for b in boundaries]), np.asarray([b.radius for b in boundaries]),
        np.asarray([b.rate for b in boundaries]), np.full(len(obs),cfg.obstacle_acceleration_bound))


def initialization_report(robot, obstacles, boundaries, cfg, tolerance=1e-10):
    if tolerance < 0:
        raise ValueError('tolerance must be nonnegative.')
    if len(obstacles) != len(boundaries) or not len(boundaries):
        raise ValueError('Require at least one obstacle and one boundary per obstacle.')
    rows=[]
    for i,(o,b) in enumerate(zip(obstacles,boundaries)):
        if b.base <= 0 or b.maximum <= b.base:
            raise ValueError('Boundary interval must be positive and nondegenerate.')
        v=evaluate(robot,o,b,cfg);theta=b.radius+b.rate/cfg.omega
        _,ref,base=reference_radius(v['q'],float(o[9]),cfg)
        consistent=(abs(b.base-base)<=tolerance and abs(b.maximum-(base+cfg.radius_span))<=tolerance)
        interval=(b.base-tolerance<=b.radius<=b.maximum+tolerance and b.base-tolerance<=theta<=b.maximum+tolerance)
        reference=(b.base-tolerance<=b.reference<=b.maximum+tolerance and abs(b.reference-ref)<=tolerance)
        rows.append(dict(obstacle=i, s=float(v['s']), h=float(v['h']),
            radius_interval_and_theta=bool(interval), reference_matches_current_law=bool(reference),
            radius_parameters_consistent=bool(consistent),
            initial_s_h_nonnegative=bool(v['s']>=-tolerance and v['h']>=-tolerance)))
    return dict(passed=all(r['radius_interval_and_theta'] and r['reference_matches_current_law'] and
        r['radius_parameters_consistent'] and r['initial_s_h_nonnegative'] for r in rows),
        rows=rows, floating_point_tolerance=tolerance,
        scope='Initialization only; neither ongoing feasibility nor goal arrival follows from this check.')


def current_common_set(B,C,limits=(2.,2.,2.),margins=None,tolerance=1e-8):
    """max_u min_i(B_i.u+C_i-margin_i), with the declared input box.

    Positive margins describe sampling tightening. An LP numerical failure is
    `solver_error`, never silently classified as physical infeasibility.
    """
    B=np.asarray(B,dtype=float);C=np.asarray(C,dtype=float);limits=np.asarray(limits,dtype=float)
    if B.ndim!=2 or B.shape[1]!=3 or C.shape!=(len(B),) or not len(B):
        raise ValueError('Require at least one affine obstacle constraint.')
    if limits.shape!=(3,) or not np.isfinite(limits).all() or np.any(limits<=0) or tolerance<0:
        raise ValueError('Require three positive finite box limits and nonnegative tolerance.')
    margin=np.zeros(len(B)) if margins is None else np.asarray(margins,dtype=float)
    if margin.shape!=C.shape or np.any(margin<0):raise ValueError('Invalid tightening margins.')
    if not np.isfinite(B).all() or not np.isfinite(C).all():raise ValueError('Nonfinite affine coefficients.')
    if not np.isfinite(margin).all():
        return dict(status='invalid_interval_bound',feasible=None,margin=None,witness=None,
                    scope='The requested sampling interval has no finite certified bound; this does not classify continuous K.')
    result=linprog([0.,0.,0.,-1.],A_ub=np.c_[-B,np.ones(len(B))],b_ub=C-margin,
        bounds=[(-x,x) for x in limits]+[(None,None)],method='highs')
    if not result.success:return dict(status='solver_error',feasible=None,margin=None,witness=None,solver_message=result.message)
    witness=np.clip(result.x[:3],-limits,limits);value=float(np.min(B@witness+C-margin))
    if value>=0:status='feasible';feasible=True
    elif value < -tolerance:status='infeasible';feasible=False
    else:status='numerical_boundary';feasible=None
    return dict(status=status,feasible=feasible,margin=value,witness=witness.tolist(),
                individual_box_margins=(C-margin+np.abs(B)@limits).tolist(),
                scope='Sampling-tightened current set' if margins is not None else 'Continuous-time current set')


def initial_common_margin(robot,obstacles,boundaries,cfg,witness=None,verification_margin=1e-9):
    """Proposition (a): a robust *fixed-input* common witness and strict δ0.

    For each state, ao is removed algebraically from A. The bound therefore
    does not use the current acceleration sample as a future prediction.
    """
    if verification_margin<0:raise ValueError('verification_margin must be nonnegative.')
    init=initialization_report(robot,obstacles,boundaries,cfg)
    values=[evaluate(robot,o,b,cfg) for o,b in zip(obstacles,boundaries)]
    B=np.array([v['B'] for v in values]);C=[]
    for v,o in zip(values,obstacles):
        A0=v['A']+v['B']@o[6:9]
        C.append(A0+cfg.cbf_gain*v['h']-cfg.obstacle_acceleration_bound*np.linalg.norm(v['B']))
    C=np.array(C);limits=np.array(cfg.acceleration_limit)
    optimized=None
    if witness is None:
        optimized=current_common_set(B,C,limits)
        witness=optimized['witness']
        if witness is None:return dict(status='solver_error',certified=False,initialization=init,solver=optimized)
    u=_array(witness,(3,),'witness');inside=bool(np.all(np.abs(u)<=limits))
    Gi=B@u+C;delta=float(Gi.min())
    # The proposition's feasibility result does not require s,h>=0. Keep that
    # condition separate, because it is additionally needed for distance safety.
    radius_ok=all(r['radius_interval_and_theta'] and r['reference_matches_current_law'] and
                  r['radius_parameters_consistent'] for r in init['rows'])
    passed=bool(inside and radius_ok and delta>verification_margin)
    return dict(status='strict_initial_margin' if passed else 'not_certified',certified=passed,
        witness=u.tolist(),Gi=Gi.tolist(),delta0=delta,input_within_box=inside,
        radius_initialization_valid=radius_ok,distance_safety_initialization_valid=init['passed'],
        initialization=init,verification_margin=verification_margin,
        certified_horizon_s=None,
        scope='Positive δ0 establishes existence of a short interval. No numeric T0 is claimed without verified local bounds.')


def local_interval_from_verified_bounds(delta0,rho,L,M):
    """Paper's T0 formula. Caller must independently justify the supplied bounds."""
    if not all(np.isfinite(v) and v>0 for v in (delta0,rho,L,M)):
        raise ValueError('Require positive finite delta0,rho,L,M.')
    return dict(T0=float(min(rho/(2*M),delta0/(2*L*M))),
                scope='Conditional on independently valid local L,M,rho; this helper does not verify those bounds.')


def finite_interval_terms(state,cfg,T):
    """Every quantity in Proposition (b), using initial data only."""
    if not np.isfinite(T) or T<0:raise ValueError('T must be finite and nonnegative.')
    D=np.linalg.norm(state.relative_position,axis=1);e0=state.relative_position/D[:,None]
    speed=np.linalg.norm(state.relative_velocity,axis=1)
    aU=float(np.linalg.norm(cfg.acceleration_limit));c=aU+state.obstacle_acceleration_bound
    C=np.full(len(D),2*cfg.omega**2*cfg.radius_span)
    W=speed+c*T;b=speed*T+.5*c*T*T
    sminus=D-b-state.base_radius-.5*C*T*T;splus=D+b-state.base_radius
    kappabar=1/cfg.clearance_time+cfg.beta*W*W;E=2*b/D
    offset=(cfg.cbf_gain/cfg.clearance_time*sminus-C
            -(kappabar+cfg.cbf_gain)*(W+C*T)-2*cfg.beta*W**3
            -state.obstacle_acceleration_bound*np.sqrt(1+4*cfg.beta**2*splus**2*W**2))
    norm_coefficient=E+2*cfg.beta*splus*W
    return dict(D=D,e0=e0,aU=aU,c=c,C=C,W=W,b=b,s_minus=sminus,s_plus=splus,
                kappa_bar=kappabar,E=E,offset=offset,norm_coefficient=norm_coefficient)


def finite_interval_check(state,cfg,T,witness=None,verification_margin=1e-9):
    """Find one ub for the WHOLE interval, then recheck explicit inequalities.

    Finding ub is a small SOCP; it is offline/certificate work, not part of Γ or
    the acceleration QP. A fixed supplied witness needs only NumPy.
    """
    if not np.isfinite(T) or T<=0:raise ValueError('The finite-interval proposition requires T>0.')
    if not np.isfinite(verification_margin) or verification_margin<0:raise ValueError('verification_margin must be finite and nonnegative.')
    t=finite_interval_terms(state,cfg,T);limits=np.asarray(cfg.acceleration_limit)
    init_ok=bool(np.array_equal(state.radius,state.base_radius) and np.all(state.rate==0))
    result=dict(status='not_certified',certified=False,T=float(T),
        radius_initialization_is_R0_zero_rate=init_ok,reset_radius=False,
        s_minus=t['s_minus'].tolist(),verification_margin=verification_margin,
        scope='Continuous-time K on [0,T], not the sampling-tightened set; no claim of obstacle passage by T.')
    if not init_ok:
        result['reason']='Proposition (b) requires actual initial R=R0 and Rdot=0; states were not reset.';return result
    if np.min(t['s_minus'])<verification_margin:
        result['reason']='The conservative s_minus condition is not met; current QP feasibility is undecided.';return result
    if np.any(t['norm_coefficient']<0):raise ValueError('Certificate domain requires nonnegative norm coefficients.')
    if witness is None:
        import cvxpy as cp
        u=cp.Variable(3);norm_u=cp.Variable(nonneg=True);delta=cp.Variable()
        problem=cp.Problem(cp.Maximize(delta),[u<=limits,u>=-limits,cp.norm(u,2)<=norm_u,
            t['e0']@u+cp.multiply(t['norm_coefficient'],norm_u)+delta<=t['offset']])
        try:problem.solve(solver='CLARABEL')
        except cp.error.SolverError as exc:
            result.update(status='solver_error',reason=str(exc));return result
        result['solver_status']=str(problem.status)
        if u.value is None:
            result.update(status='solver_error',reason='Certificate witness optimization returned no input.');return result
        witness=np.clip(np.asarray(u.value).ravel(),-limits,limits)
    u=_array(witness,(3,),'witness');inside=bool(np.all(np.abs(u)<=limits))
    lower=t['offset']-t['e0']@u-t['norm_coefficient']*np.linalg.norm(u)
    passed=bool(inside and lower.min()>=verification_margin)
    result.update(status='certified' if passed else 'not_certified',certified=passed,
        witness=u.tolist(),input_within_box=inside,G_lower=lower.tolist(),
        minimum_G_lower=float(lower.min()),
        reason='All explicit sufficient inequalities passed.' if passed else
               'Sufficient inequalities not met; this does not prove the QP infeasible.')
    return result


def common_direction_check(barriers,cfg,direction,H,obstacle_bounds=None,tolerance=1e-10):
    """Theorem 2 with an explicit common direction, not individual feasibility."""
    d=_array(direction,(3,),'direction')
    if abs(np.linalg.norm(d)-1)>tolerance:raise ValueError('direction must be unit length.')
    B=np.asarray([v['B'] for v in barriers]);n=len(B)
    H=_array(H,(n,),'H')
    bounds=np.full(n,cfg.obstacle_acceleration_bound) if obstacle_bounds is None else _array(obstacle_bounds,(n,),'bounds')
    if not n or np.any(H<0) or np.any(bounds<0) or np.any(np.linalg.norm(B,axis=1)==0):
        raise ValueError('Require nonempty nonzero coefficients and nonnegative H/acceleration bounds.')
    cosine=(-B@d)/np.linalg.norm(B,axis=1);minimum=float(cosine.min())
    margin=float(min(cfg.acceleration_limit)*minimum-np.max(bounds+H))
    return dict(certified=bool(minimum>0 and margin>=0),minimum_cosine=minimum,
                acceleration_margin=margin,witness=(-min(cfg.acceleration_limit)*d).tolist(),
                scope='Current common-direction sufficient check; requires all declared state-domain hypotheses separately.')
