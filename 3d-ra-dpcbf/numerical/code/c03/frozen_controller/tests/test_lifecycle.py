"""Regression checks for the actual previous circling fault and switching rules."""
import numpy as np
import pytest
from configuration import load_config
from detour import Detour


def fixture(fixed=False):
    cfg=load_config();engine=Detour(cfg['detour'],fixed_direction=fixed)
    engine.j=0;engine.d=np.array([0.,1.,0.]);engine.e=np.array([1.,0.,0.])
    x=np.array([0.,0.,0.,-1.,0.,0.]);goal=np.array([-8.,0.,0.])
    obs=np.array([[2.,0.,0.,0.,0.,0.,0.,0.,0.,.5],[-4.,0.,0.,0.,0.,0.,0.,0.,0.,.5]])
    log=np.zeros((2,16));log[:,0]=[2,4];log[:,1]=[1,-1];log[:,3]=1.;log[:,6]=[1,3]
    B=np.array([[1.,0.,0.]]);C=np.array([0.])
    return cfg,engine,x,goal,obs,log,B,C


@pytest.mark.parametrize('fixed',[False,True])
def test_original_release_rule_does_not_ignore_other_obstacle_or_unsafe_nominal(fixed):
    cfg,m,x,goal,obs,log,B,C=fixture(fixed)
    nominal=np.array([-1.,0.,0.])
    for _ in range(10):
        z=m.update(x,goal,obs,log,B,C,nominal,.1,certified=False)
        assert z['active'] and z['off_elapsed']==0 and m.j==0
        assert z['primary_clear'] and not z['goal_clear'] and not z['nominal_admissible']


def clear_fixture(fixed=False):
    cfg,m,x,goal,obs,log,B,C=fixture(fixed)
    obs[1,1]=4. # Move only this unit-test fixture off the goal segment.
    return cfg,m,x,goal,obs,log,B,C


@pytest.mark.parametrize('fixed',[False,True])
def test_all_original_release_conditions_confirm_for_full_dwell(fixed):
    cfg,m,x,goal,obs,log,B,C=clear_fixture(fixed)
    outputs=[m.update(x,goal,obs,log,B,C,np.zeros(3),.1,certified=False) for _ in range(5)]
    assert all(z['active'] for z in outputs[:4])
    assert not outputs[4]['active'] and outputs[4]['event']==2 and m.j==-1
    assert outputs[4]['reason']=='goal_clear_primary_receding_nominal_safe'
    assert abs(outputs[4]['off_elapsed']-.4)<1e-12
    assert outputs[4]['goal_clear'] and outputs[4]['nominal_admissible']


@pytest.mark.parametrize('interruption',['receding','nominal','occlusion'])
def test_confirmation_timer_resets_after_any_original_predicate_is_interrupted(interruption):
    cfg,m,x,goal,obs,log,B,C=clear_fixture();nom=np.zeros(3)
    for _ in range(3):m.update(x,goal,obs,log,B,C,nom,.1,certified=False)
    assert abs(m.off_time-.2)<1e-12
    if interruption=='receding':x[3]=1.
    if interruption=='nominal':nom[0]=-1.
    if interruption=='occlusion':obs[1,1]=0.
    m.update(x,goal,obs,log,B,C,nom,.1,certified=False)
    assert m.off_time==0
    x[3]=-1.;nom[:]=0.;obs[1,1]=4.
    z=m.update(x,goal,obs,log,B,C,nom,.1,certified=False)
    assert z['active'] and z['off_elapsed']==0


def test_held_direction_is_not_reselected_while_capacity_is_sufficient():
    cfg,m,x,goal,obs,log,B,C=fixture();goal=np.array([8.,0.,0.])
    def forbidden(*args):raise AssertionError('Unnecessary direction reselection')
    m.choose=forbidden;before=m.d.copy()
    z=m.update(x,goal,obs,log,B,C,np.zeros(3),.1,certified=False)
    assert z['active'];np.testing.assert_allclose(z['d'],before)


def test_new_direction_recomputes_actual_projection_instead_of_zeroing_chi():
    cfg,m,x,goal,obs,log,B,C=fixture();m.j=-1;goal=np.array([8.,0.,0.]);x[4]=.4
    chosen=np.array([0.,-1.,0.]);m.choose=lambda *args:(chosen,2.)
    z=m.update(x,goal,obs,log,B,C,np.zeros(3),.1,certified=False)
    assert z['event']==1 and abs(z['chi']+.4)<1e-12


def test_capacity_loss_pauses_auxiliary_constraint():
    cfg,m,x,goal,obs,log,B,C=fixture();goal=np.array([8.,0.,0.])
    # Force u_y=0 and u_z=0; the held direction has no positive startup capacity.
    B=np.array([[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    C=np.zeros(4);z=m.update(x,goal,obs,log,B,C,np.zeros(3),.1,certified=False)
    assert not z['active'] and z['event']==3 and m.j==-1


def test_empty_base_set_never_starts_direction_search():
    cfg,m,x,goal,obs,log,B,C=fixture();m.choose=lambda *a:(_ for _ in ()).throw(AssertionError('searched empty K'))
    z=m.update(x,goal,obs,log,np.array([[1.,0,0],[-1.,0,0]]),np.array([-1.,-1.]),
        np.zeros(3),.1,certified=False)
    assert not z['active'] and z['reason']=='base_set_infeasible'


def test_fixed_strength_ablation_pauses_without_corrupting_primary_safety_set():
    cfg,m,x,goal,obs,log,B,C=fixture();m.fixed_strength=True;m.j=-1;goal=np.array([8.,0.,0.])
    # Startup is possible, but the fixed .3 request exceeds the .2 directional capacity.
    B=np.array([[0.,1.,0.],[0.,-1.,0.],[0.,0.,1.],[0.,0.,-1.]])
    C=np.array([.2,.2,0.,0.]);before_B=B.copy();before_C=C.copy()
    z=m.update(x,goal,obs,log,B,C,np.zeros(3),.1,certified=False)
    assert not z['active'] and z['reason']=='fixed_strength_unexecutable'
    np.testing.assert_array_equal(B,before_B);np.testing.assert_array_equal(C,before_C)
