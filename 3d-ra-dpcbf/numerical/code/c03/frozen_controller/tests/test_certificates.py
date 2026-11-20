"""Independent algebra/optimization checks; no navigation experiments."""
from dataclasses import replace
import numpy as np
import pytest
from configuration import load_config,core_config
from ra_dpcbf.core import Boundary,evaluate,boundary_step,initialize_boundary
from ra_dpcbf.certificates import (initial_state,initialization_report,initial_common_margin,
    finite_interval_terms,finite_interval_check,current_common_set,local_interval_from_verified_bounds)


def far_state():
    cfg=core_config(load_config());x=np.zeros(6)
    o=np.array([[40.,0.,0.,0.,0.,0.,0.,0.,0.,.5],
                [0.,40.,0.,0.,0.,0.,0.,0.,0.,.5]])
    b=[initialize_boundary(x,v,cfg) for v in o]
    return cfg,x,o,b


def test_scalar_common_margin_has_analytical_box_solution():
    cfg,x,o,b=far_state();o=o[:1];b=b[:1]
    report=initial_common_margin(x,o,b,cfg)
    value=evaluate(x,o[0],b[0],cfg)
    expected=value['A']+cfg.cbf_gain*value['h']+2-cfg.obstacle_acceleration_bound
    assert report['certified'] and report['certified_horizon_s'] is None
    assert abs(report['delta0']-expected)<1e-9


def test_initial_robust_margin_does_not_depend_on_current_acceleration_sample():
    cfg,x,o,b=far_state();u=np.array([-.4,-.3,.1])
    a=initial_common_margin(x,o,b,cfg,witness=u)
    changed=o.copy();changed[:,6:9]=[[.3,-.2,.1],[-.2,.4,.1]]
    c=initial_common_margin(x,changed,b,cfg,witness=u)
    np.testing.assert_allclose(a['Gi'],c['Gi'],atol=1e-10)
    # The worst permitted acceleration reaches the robust lower bound exactly.
    for i,bi in enumerate(b):
        v=evaluate(x,o[i],bi,cfg);worst=o[i].copy()
        worst[6:9]=cfg.obstacle_acceleration_bound*v['B']/np.linalg.norm(v['B'])
        w=evaluate(x,worst,bi,cfg)
        assert abs(w['A']+w['B']@u+cfg.cbf_gain*w['h']-a['Gi'][i])<1e-9


def test_initialization_rejects_bad_theta_without_changing_boundary():
    cfg,x,o,b=far_state();bad=replace(b[0],rate=cfg.omega*(cfg.radius_span+.2))
    report=initialization_report(x,o,[bad,b[1]],cfg)
    assert not report['passed'] and not report['rows'][0]['radius_interval_and_theta']
    assert bad.rate==cfg.omega*(cfg.radius_span+.2)


def test_individual_feasibility_does_not_imply_common_feasibility():
    result=current_common_set([[1.,0,0],[-1.,0,0]],[-1.,-1.])
    assert min(result['individual_box_margins'])==1.
    assert result['status']=='infeasible' and abs(result['margin']+1)<1e-12


def test_sampling_set_is_separate_from_continuous_set():
    B=np.array([[1.,0.,0.]]);C=np.array([0.])
    assert current_common_set(B,C)['feasible']
    assert current_common_set(B,C,margins=np.array([2.1]))['feasible'] is False
    assert current_common_set(B,C,margins=np.array([np.inf]))['status']=='invalid_interval_bound'


def test_finite_interval_formula_and_norm_use_box_circumscribed_radius():
    cfg,x,o,b=far_state();state=initial_state(x,o,b,cfg);T=.05
    terms=finite_interval_terms(state,cfg,T)
    c=np.sqrt(12)+.8;C=2*cfg.omega**2*cfg.radius_span
    W=c*T;bound=.5*c*T*T;sminus=40-bound-.95-.5*C*T*T;splus=40+bound-.95
    u=np.array([-.4,-.3,.2]);E=2*bound/40
    manual=(cfg.cbf_gain/cfg.clearance_time*sminus-C
        -(1/cfg.clearance_time+cfg.beta*W*W+cfg.cbf_gain)*(W+C*T)
        -2*cfg.beta*W**3-u[0]-np.linalg.norm(u)*(E+2*cfg.beta*splus*W)
        -.8*np.sqrt(1+4*cfg.beta**2*splus*splus*W*W))
    report=finite_interval_check(state,cfg,T,witness=u)
    assert abs(terms['aU']-np.sqrt(12))<1e-12
    assert abs(report['G_lower'][0]-manual)<1e-10 and report['certified']


def test_optimized_witness_is_rechecked_and_failure_is_not_infeasibility():
    cfg,x,o,b=far_state();state=initial_state(x,o,b,cfg)
    good=finite_interval_check(state,cfg,.05)
    assert good['certified']
    direct=finite_interval_check(state,cfg,.05,witness=good['witness'])
    assert direct['certified'] and abs(direct['minimum_G_lower']-good['minimum_G_lower'])<1e-10
    bad=finite_interval_check(state,cfg,10.)
    assert not bad['certified'] and bad['status']=='not_certified'
    values=[evaluate(x,oo,bb,cfg) for oo,bb in zip(o,b)]
    assert current_common_set(np.array([v['B'] for v in values]),
        np.array([v['A']+cfg.cbf_gain*v['h'] for v in values]))['feasible']


def test_finite_interval_never_resets_nonzero_radius_state():
    cfg,x,o,b=far_state();b[0]=replace(b[0],radius=b[0].radius+.1)
    report=finite_interval_check(initial_state(x,o,b,cfg),cfg,.01)
    assert not report['certified'] and report['reset_radius'] is False
    assert abs(b[0].radius-.95-.1)<1e-12


def test_lower_bound_covers_admissible_constant_acceleration_states():
    cfg,x,o,b=far_state();state=initial_state(x,o,b,cfg);T=.05;ub=np.array([-.4,-.3,.1])
    lower=np.array(finite_interval_check(state,cfg,T,witness=ub)['G_lower'])
    rng=np.random.default_rng(315)
    # Analytical constant-acceleration states, not a controller rollout.
    for _ in range(35):
        t=rng.uniform(0,T);u=rng.uniform(-2,2,3);ao=rng.normal(size=(len(o),3))
        ao*=rng.uniform(0,.8,size=(len(o),1))/np.linalg.norm(ao,axis=1)[:,None]
        xx=np.r_[.5*t*t*u,t*u];oo=o.copy();oo[:,:3]+=.5*t*t*ao;oo[:,3:6]+=t*ao;oo[:,6:9]=ao
        for i,bb in enumerate(b):
            v=evaluate(xx,oo[i],boundary_step(bb,t,cfg),cfg)
            A0=v['A']+v['B']@ao[i]
            G=A0+v['B']@ub+cfg.cbf_gain*v['h']-.8*np.linalg.norm(v['B'])
            assert G>=lower[i]-1e-9


def test_local_interval_requires_explicit_validated_bounds():
    r=local_interval_from_verified_bounds(2.,.4,3.,5.)
    assert abs(r['T0']-min(.4/10,2/30))<1e-12
    with pytest.raises(ValueError):local_interval_from_verified_bounds(0.,.4,3.,5.)
