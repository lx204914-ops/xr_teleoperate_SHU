"""Independent checks of the new mechanism, rather than implementation mirrors."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.optimize import linprog, minimize
from detour import vertices,polytope,Detour,support,transport
from scene_tapes import state_at,load_scene
from dynamic_study.kernel import constraints,project_qp
from ra_dpcbf.core import Config,Boundary,evaluate,los

def test_vertex_direction_against_dense_angular_search():
    rng=np.random.default_rng(718);e=np.array([1.,1.,.4]);e/=np.linalg.norm(e)
    n1=np.cross(e,[0.,0.,1.]);n1/=np.linalg.norm(n1);n2=np.cross(e,n1)
    z=.25*n1-.1*n2;ao=np.array([.2,-.1,.1]);q=-1.;D=3.
    B=rng.normal(size=(7,3));C=np.ones(7)*1.2;A,b=polytope(B,C)
    engine=Detour({'k_perp':1.2});direction,capacity=engine.choose(A,b,e,z,ao,q,D)
    grid=[]
    for theta in np.linspace(0,2*np.pi,241):
        d=np.cos(theta)*n1+np.sin(theta)*n2
        result=linprog(-d,A_ub=-A,b_ub=-b,bounds=[(None,None)]*3,method='highs')
        grid.append(d@result.x-d@ao-(1.2-q/D)*(d@z))
    assert max(grid)<=capacity+1e-7
    assert capacity-max(grid)<.002
    assert abs(direction@e)<1e-10 and abs(np.linalg.norm(direction)-1)<1e-10

def test_affine_derivative_matches_independent_difference():
    cfg=Config();x=np.array([1.,2.,3.,.8,.1,-.2]);o=np.array([4.,3.,3.3,-.4,.3,.2,.1,-.15,.2,.5])
    boundary=Boundary(1.1,.04,1.3,.95,1.8087176689796248);u=np.array([-.3,.4,.2])
    values=evaluate(x,o,boundary,cfg);A,B=values['A'],values['B']
    def value_at(h):
        xx=x+np.r_[x[3:]*h+.5*u*h*h,u*h]
        oo=o.copy();oo[:3]+=o[3:6]*h+.5*o[6:9]*h*h;oo[3:6]+=o[6:9]*h
        bb=Boundary(boundary.radius+h*boundary.rate+.5*h*h*boundary.acceleration(cfg),
                    boundary.rate+h*boundary.acceleration(cfg),boundary.reference,boundary.base,boundary.maximum)
        return evaluate(xx,oo,bb,cfg)['h']
    numerical=(value_at(1e-6)-value_at(-1e-6))/2e-6
    assert abs(numerical-(A+B@u))<1e-7

def test_transport_and_chi_dynamics():
    r=np.array([3.,1.,.5]);w=np.array([-.8,.3,.1]);a=np.array([.2,-.1,.15]);D=np.linalg.norm(r);e=r/D
    d=np.cross(e,[0.,0.,1.]);d/=np.linalg.norm(d);q=e@w;chi=-d@w;dt=1e-7
    rr=r+dt*w+.5*dt**2*a;ee=rr/np.linalg.norm(rr);dd=transport(d,e,ee)
    numerical=(-dd@(w+dt*a)-chi)/dt
    # a is relative acceleration ao-u, hence d*(u-ao)=-d*a.
    assert abs(numerical-(-d@a-q/D*chi))<2e-7
    assert abs(dd@ee)<1e-12 and abs(np.linalg.norm(dd)-1)<1e-12

def test_augmented_qp_against_scipy():
    rng=np.random.default_rng(91)
    for _ in range(12):
        B=rng.normal(size=(10,3));C=rng.uniform(.1,1.2,10);target=rng.normal(size=3)*2
        A,b=polytope(B,C);d=rng.normal(size=3);d/=np.linalg.norm(d);maximum,_=support(A,b,d)
        lower=.6*maximum
        BB=np.r_[B,d[None,:]];CC=np.r_[C,-lower]
        u,ok,_=project_qp(target,BB,CC)
        result=minimize(lambda v:.5*np.sum((v-target)**2),np.zeros(3),jac=lambda v:v-target,
          bounds=[(-2,2)]*3,constraints=[{'type':'ineq','fun':lambda v:BB@v+CC,'jac':lambda v:BB}],method='SLSQP',
          options={'ftol':1e-11,'maxiter':500})
        assert ok and result.success and np.linalg.norm(u-result.x)<1e-5

def test_reference_knots_are_retained_and_bounded_tape_is_kinematic():
    reference=load_scene('default8','reference',duration=1.)
    for k in [0,1,17,101,len(reference['tape'])-1]:
        out=state_at(reference['tape'],k*reference['tape_dt'],reference['tape_dt'],reference['mode'])
        assert np.max(np.abs(out-reference['tape'][k]))<1e-8
    bounded=load_scene('default8','bounded',duration=1.)
    tape=bounded['tape'];td=bounded['tape_dt'];t=.173;eps=1e-6
    before=state_at(tape,t-eps,td,1);now=state_at(tape,t,td,1);after=state_at(tape,t+eps,td,1)
    assert np.max(np.abs((after[:,:3]-before[:,:3])/(2*eps)-now[:,3:6]))<1e-7
    assert np.max(np.abs((after[:,3:6]-before[:,3:6])/(2*eps)-now[:,6:9]))<1e-7
    assert np.linalg.norm(tape[:,:,6:9],axis=2).max()<=.7200001
    assert np.linalg.norm(np.diff(tape[:,:,6:9],axis=0)/td,axis=2).max()<=1.1000001
