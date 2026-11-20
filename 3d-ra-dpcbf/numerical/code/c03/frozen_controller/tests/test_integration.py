"""State-level integration tests; do not invoke run_suite.run or renderers."""
import numpy as np
from configuration import load_config,core_config
from dynamic_study.kernel import constraints,radius_ref
from ra_dpcbf.core import evaluate,initialize_boundary
from safety_filter import SafetyFilter
from feasibility_checks import check_initial_state


def state():
    config=load_config();p=np.array(config['parameters']);x=np.array([0.,0.,3.,1.,0.,0.])
    obs=np.array([[4.,0.,3.,0.,0.,0.,0.,0.,0.,.5]])
    R=p[21]+p[22]+obs[:,9];V=np.zeros(1);ref=np.array([radius_ref(-1.,R[0],p)[0]])
    B,C,log=constraints(x,obs,R,V,ref,p,5)
    return config,p,x,obs,R,V,ref,B,C,log


def test_active_config_and_core_and_fast_kernel_agree():
    config,p,x,obs,R,V,ref,B,C,log=state();cfg=core_config(config)
    b=initialize_boundary(x,obs[0],cfg);v=evaluate(x,obs[0],b,cfg)
    assert abs(v['Rref']-ref[0])<1e-12 and abs(v['h']-log[0,7])<1e-12
    np.testing.assert_allclose(v['B'],B[0],atol=1e-12)
    assert abs(v['A']+cfg.cbf_gain*v['h']-C[0])<1e-12


def test_zero_transverse_speed_starts_executable_motion_with_all_hard_constraints():
    config,p,x,obs,R,V,ref,B,C,log=state();goal=np.array([10.,0.,3.])
    filt=SafetyFilter(config);res=filt.solve(x,goal,obs,log,B,C,np.zeros(3),np.zeros(3),
                                         config['control_dt'],certified=False)
    assert res.status=='ok' and res.detour['active'] and res.detour['chi']==0
    assert np.min(B@res.control+C)>=-1e-8
    d=res.detour['d'];assert abs(d[0])<1e-12
    chi_dot=d@(res.control-obs[0,6:9])
    assert chi_dot>=res.detour['strength']-1e-8
    assert res.auxiliary_changed_input and res.auxiliary_binding


def test_no_detour_uses_identical_primary_safety_filter():
    config,p,x,obs,R,V,ref,B,C,log=state()
    off=SafetyFilter(config,'no_detour').solve(x,np.array([10.,0.,3.]),obs,log,B,C,
        np.zeros(3),np.zeros(3),config['control_dt'],certified=False)
    assert off.status=='ok' and not off.detour['active']
    assert abs(off.control[1])+abs(off.control[2])<1e-12
    assert np.min(B@off.control+C)>=-1e-8


def test_primary_infeasibility_and_interval_bound_failure_are_distinct():
    config,p,x,obs,R,V,ref,B,C,log=state();filt=SafetyFilter(config)
    bad=filt.solve(x,np.array([10.,0,3]),obs,log,np.array([[1.,0,0]]),np.array([-3.]),
        np.zeros(3),np.zeros(3),config['control_dt'],certified=False)
    assert bad.status=='base_qp_infeasible' and bad.control is None
    no_bound=filt.solve(x,np.array([10.,0,3]),obs,log,B,C,np.zeros(3),np.zeros(3),
        config['control_dt'],margins=np.array([np.inf]))
    assert no_bound.status=='invalid_sampling_bound' and no_bound.control is None


def test_certificate_failure_does_not_disable_feasible_filter():
    config,p,x,obs,R,V,ref,B,C,log=state()
    report=check_initial_state(x,obs,R,V,ref,config,horizon=10.)
    assert report['initialization']['passed'] and not report['finite_interval']['certified']
    result=SafetyFilter(config).solve(x,np.array([10.,0,3]),obs,log,B,C,np.zeros(3),
        np.zeros(3),config['control_dt'],certified=False)
    assert result.status=='ok'


def test_observed_acceleration_bound_violation_is_not_certified_or_silently_ignored():
    config,p,x,obs,R,V,ref,B,C,log=state();obs[0,6]=.9
    result=SafetyFilter(config).solve(x,np.array([10.,0,3]),obs,log,B,C,np.zeros(3),
        np.zeros(3),config['control_dt'],certified=True)
    assert result.status=='obstacle_acceleration_bound_violation' and result.control is None


def test_sampling_infeasibility_is_not_reported_as_empty_continuous_set():
    config,p,x,obs,R,V,ref,B,C,log=state()
    result=SafetyFilter(config).solve(x,np.array([10.,0,3]),obs,log,np.array([[1.,0,0]]),
        np.zeros(1),np.zeros(3),np.zeros(3),config['control_dt'],margins=np.array([2.1]),certified=False)
    assert result.status=='sampling_set_infeasible' and result.control is None


def test_solver_failure_is_not_treated_as_method_infeasibility(monkeypatch):
    import safety_filter
    config,p,x,obs,R,V,ref,B,C,log=state()
    monkeypatch.setattr(safety_filter,'project_qp',lambda *a:(np.zeros(3),False,0))
    result=SafetyFilter(config).solve(x,np.array([10.,0,3]),obs,log,B,C,np.zeros(3),
        np.zeros(3),config['control_dt'],certified=False)
    assert result.status=='base_qp_numerical_failure' and result.control is None
