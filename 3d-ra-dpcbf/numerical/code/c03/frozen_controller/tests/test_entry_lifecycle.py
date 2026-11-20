"""Frozen-scene binding and original lifecycle contract regression checks."""
from pathlib import Path
import hashlib
import numpy as np
import pytest
from configuration import load_config
from scene_tapes import load_scene,state_at
from safety_filter import SafetyFilter
from detour import Detour
from test_integration import state
from test_lifecycle import fixture


def test_corridor_loader_keeps_the_previous_final_scene_byte_identical():
    a=load_scene('corridor_crossing','bounded',seed=1,duration=.01)
    b=load_scene('corridor_crossing','bounded',seed=999,duration=24.)
    expected='053e9c8c2ec575bab61a4928113cfb1cb58c7c52d118ce0c102d28dedbd2e8b3'
    assert hashlib.sha256(a['tape'].tobytes()).hexdigest()==expected
    np.testing.assert_array_equal(a['tape'],b['tape'])
    assert a['revision']==4 and a['tape'].shape==(2401,4,10)
    assert a['manoeuvre_seed'] is None
    np.testing.assert_array_equal(a['start'],[0,0,3,0,0,0])
    np.testing.assert_array_equal(a['goal'],[15.5,0,3])


def test_corridor_motion_derivatives_and_bounds_with_no_robot_feedback():
    scene=load_scene('corridor_crossing');tape=scene['tape'];dt=scene['tape_dt']
    assert np.linalg.norm(tape[:,:,6:9],axis=2).max()<.8
    assert np.linalg.norm(np.diff(tape[:,:,6:9],axis=0)/dt,axis=2).max()<1.1
    # Constant nonzero x velocity proves continuous motion even at lateral turns.
    np.testing.assert_allclose(tape[:,:,3],np.broadcast_to([-.12,.28,-.28,-.15],tape[:,:,3].shape))
    for t in [.137,4.323,8.917,20.103]:
        eps=1e-6;before=state_at(tape,t-eps,dt,1);now=state_at(tape,t,dt,1);after=state_at(tape,t+eps,dt,1)
        np.testing.assert_allclose((after[:,:3]-before[:,:3])/(2*eps),now[:,3:6],atol=2e-8,rtol=0)
        np.testing.assert_allclose((after[:,3:6]-before[:,3:6])/(2*eps),now[:,6:9],atol=2e-8,rtol=0)


def test_reference_mode_cannot_mislabel_the_frozen_bounded_scene():
    with pytest.raises(ValueError,match='bounded'):load_scene('corridor_crossing','reference')


def test_nominal_release_check_uses_raw_K_instead_of_sampling_tightening():
    cfg,p,x,obs,R,V,ref,B,C,log=state()
    # The raw nominal is exactly admissible but excluded by the held-sample margin.
    B=np.array([[1.,0.,0.]]);C=np.zeros(1);nominal=np.zeros(3)
    result=SafetyFilter(cfg).solve(x,np.array([10.,0.,3.]),obs,log,B,C,nominal,np.zeros(3),
                                  cfg['control_dt'],margins=np.array([.2]),certified=False)
    assert result.status=='ok' and result.detour['nominal_admissible']
    assert np.min(B@nominal+C-.2)<0
    assert np.min(B@result.control+C-.2)>=-1e-7


@pytest.mark.parametrize('fixed',[False,True])
def test_capacity_loss_reselects_only_if_new_executable_direction_exists(fixed):
    cfg,m,x,goal,obs,log,B,C=fixture(fixed);goal=np.array([8.,0.,0.])
    # Old y direction has no capacity; z retains full box capacity.
    B=np.array([[0.,1.,0.],[0.,-1.,0.]]);C=np.zeros(2)
    result=m.update(x,goal,obs,log,B,C,np.zeros(3),.1,certified=False)
    if fixed:
        assert not result['active'] and result['reason']=='insufficient_executable_capacity'
    else:
        assert result['active'] and result['event']==1
        assert abs(result['d'][2])>1-1e-10 and abs(result['d'][1])<1e-10
        assert result['strength']<=result['executable_gamma']


def test_nonfinite_success_from_base_optimizer_is_a_numerical_failure(monkeypatch):
    import safety_filter
    cfg,p,x,obs,R,V,ref,B,C,log=state()
    monkeypatch.setattr(safety_filter,'project_qp',lambda *args:(np.full(3,np.nan),True,0))
    result=SafetyFilter(cfg).solve(x,np.array([10.,0,3]),obs,log,B,C,np.zeros(3),np.zeros(3),.0025,certified=False)
    assert result.status=='base_qp_numerical_failure' and result.control is None


def test_nonfinite_success_from_augmented_optimizer_is_a_numerical_failure(monkeypatch):
    import safety_filter
    cfg,p,x,obs,R,V,ref,B,C,log=state();original=safety_filter.project_qp;calls=[]
    def optimizer(*args):
        calls.append(1)
        return original(*args) if len(calls)==1 else (np.full(3,np.nan),True,0)
    monkeypatch.setattr(safety_filter,'project_qp',optimizer)
    result=SafetyFilter(cfg).solve(x,np.array([10.,0,3]),obs,log,B,C,np.zeros(3),np.zeros(3),.0025,certified=False)
    assert len(calls)==2 and result.status=='augmented_qp_numerical_failure' and result.control is None
