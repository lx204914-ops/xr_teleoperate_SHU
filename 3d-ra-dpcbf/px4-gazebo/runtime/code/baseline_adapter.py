"""Same MPC and measured-acceleration execution envelope for three baselines.
Original barrier coefficient and sample-bound modules are byte-frozen.
"""
from pathlib import Path
import sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from px4_controller import Px4Controller as DeployedController
sys.path.insert(0,str(ROOT/"frozen_baselines"))
import numpy as np
import scipy.sparse as sp
import osqp
from baseline_coefficients import coefficients
from sampled_baselines import hold_certificates
from ma_cbf_vo import terms,hard_hold,KVO
from nominal_mpc import solve
from sparse_projection import SparseProjectionQP
from ra_dpcbf.certificates import current_common_set
from certificate_pool import candidate_lower
def execution_rows(a,previous,dt):
    return np.r_[np.eye(3),-np.eye(3)],np.full(6,1.5)
class PhysicalSoftVOQP:
    def __init__(self,n,nh,smooth):
        self.n=n;self.nh=nh;self.smooth=smooth;self.nv=3+n;self.nr=nh+n+self.nv
        rows=[];cols=[]
        for j in range(3):
            rows.extend(range(nh+n));cols.extend([j]*(nh+n))
            rows.append(nh+n+j);cols.append(j)
        for i in range(n):rows.extend([nh+i,nh+n+3+i]);cols.extend([3+i]*2)
        self.rows=rows;self.cols=cols
        pattern=sp.csc_matrix((np.ones(len(rows)),(rows,cols)),shape=(self.nr,self.nv))
        self.qp=osqp.OSQP();self.qp.setup(P=sp.eye(self.nv,format="csc"),q=np.zeros(self.nv),A=pattern,
            l=np.full(self.nr,-np.inf),u=np.full(self.nr,np.inf),verbose=False,
            eps_abs=1e-9,eps_rel=1e-9,max_iter=1000,polishing=False,adaptive_rho=True,adaptive_rho_interval=25)
    def solve(self,nom,prev,B,C,VB,VC,weights):
        values=[]
        for j in range(3):values.extend(B[:,j]);values.extend(VB[:,j]);values.append(1.)
        for i in range(self.n):values.extend([1.,1.])
        A=sp.csc_matrix((np.asarray(values),(self.rows,self.cols)),shape=(self.nr,self.nv))
        lo=np.r_[-C,-VC,[-2.]*3,[0.]*self.n];hi=np.r_[[np.inf]*(self.nh+self.n),[2.]*3,[np.inf]*self.n]
        diag=np.r_[np.full(3,2*(1+self.smooth)),2*KVO*weights]
        q=np.r_[-2*(nom+self.smooth*prev),np.zeros(self.n)]
        self.qp.update(Px=diag,q=q,Ax=A.data,l=lo,u=hi)
        result=self.qp.solve(raise_error=False);sol=result.x
        def valid(y):
            return y is not None and np.isfinite(y).all() and np.min(A@y-lo)>=-2e-7 and np.min(hi-A@y)>=-2e-7
        if result.info.status_val not in (1,2) or not valid(sol):
            import clarabel
            ml=np.isfinite(lo);mh=np.isfinite(hi)
            AA=sp.vstack([-A[ml],A[mh]],format="csc");bb=np.r_[-lo[ml],hi[mh]]
            settings=clarabel.DefaultSettings();settings.verbose=False;settings.max_iter=400
            settings.tol_gap_abs=1e-10;settings.tol_gap_rel=1e-10;settings.tol_feas=1e-11
            answer=clarabel.DefaultSolver(sp.diags(diag,format="csc"),q,AA,bb,
                [clarabel.NonnegativeConeT(len(bb))],settings).solve()
            sol=np.asarray(answer.x) if str(answer.status) in ("Solved","AlmostSolved") else None
        return (sol[:3].copy(),True) if valid(sol) else (None,False)
class BaselineController(DeployedController):
    def __init__(self,method,nominal_cruise=1.5):
        super().__init__(nominal_cruise=nominal_cruise);self.method=method
        if method not in ("ma_cbf_vo","c3bf","park"):raise ValueError(method)
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs)
        self.preference_counts={"accepted":0,"hard_safety_fallback":0}
        self.preferred_backend=(PhysicalSoftVOQP(len(self.ids),len(self.ids)+15,(self.par[6])) if self.method=="ma_cbf_vo"
                                else SparseProjectionQP(len(self.ids)+15))
        self.backend=(PhysicalSoftVOQP(len(self.ids),len(self.ids)+7,(self.par[6])) if self.method=="ma_cbf_vo"
                      else SparseProjectionQP(len(self.ids)+7))
    def raw_coefficients(self,x,obs):
        if self.method=="ma_cbf_vo":
            data=terms(x,obs,self.par)
            return data[3],data[4],data[5]
        return coefficients(x,obs,self.R,self.V,self.ref,self.par,2 if self.method=="c3bf" else 8)
    def step(self,t,state,obstacles,obstacle_ids=None):
        if not self.ready or abs(float(t)-self.time)>1e-8:raise RuntimeError("Controller clock mismatch")
        x=np.asarray(state);obs=np.asarray(obstacles);tic=time.perf_counter()
        if x.shape!=(6,) or obs.shape!=(len(self.ids),10):raise ValueError("Input shape mismatch")
        if obstacle_ids is not None and tuple(obstacle_ids)!=self.ids:raise ValueError("Obstacle identity changed")
        B,C,log=self.raw_coefficients(x,obs)
        if self.last_step_time is None and np.min(log[:,7])<-1e-8:return None,"initial_safety_condition_failed",{}
        nom,plan,iterations,error=self.solve_nominal(x,self.goal,self.par[9],self.previous,self.warm,*self.mpc)
        self.warm[:-1]=plan[1:];self.warm[-1]=plan[-1]
        if self.method=="ma_cbf_vo":
            sampling,_=hard_hold(x,obs,self.par,self.config["control_dt"]);data=terms(x,obs,self.par)
        elif self.method=="park":
            sampling=np.zeros(len(obs))
        else:
            sampling,_=hold_certificates(x,obs,self.par,self.config["control_dt"],2)
        reference=self.advance_reference(nom)
        effective_nominal=reference
        tracking=.5*np.linalg.norm(B,axis=1)
        from ground_rows import rows as physical_rows
        extraB,extraC=physical_rows(self,x)
        BB=np.r_[B,extraB]
        certificate_iterations=0
        for attempt in range(8):
            CC=np.r_[C-sampling-tracking,extraC]
            if not np.isfinite(CC).all():return None,"invalid_sampling_bound",{}
            if self.method=="ma_cbf_vo":
                u,ok=self.backend.solve(effective_nominal,self.previous,BB,CC,data[0],data[1],data[6])
            else:
                smooth=self.par[6]
                target=(effective_nominal+smooth*self.previous)/(1+smooth)
                u,ok,_=self.backend.solve(target,BB,CC)
            if not ok and self.method=="park":
                from park_certificate_recovery import recover
                recovered,record=recover(self,x,obs,B,C,tracking,extraB,extraC)
                self.last_certificate_recovery=record
                if recovered is not None:
                    u=recovered;break
            if not ok:
                report=current_common_set(BB,CC,self.config["acceleration_component_limits"])
                label="candidate_certificate_set_empty" if self.method=="park" else "deployment_set_infeasible"
                return None,label if report["feasible"] is False else "qp_numerical_failure",dict(B=B.copy(),C=C.copy(),sampling_margin=sampling.copy(),tracking_reserve=tracking.copy(),ground=getattr(self,"ground_diagnostic",None))
            if np.min(BB@u+CC)<-2e-7:raise RuntimeError("Hard safety row violated")
            # No measured-acceleration preference: same pre-QP reference policy as RA.
            if self.method!="park":break
            certificate_iterations+=1
            from refine_park_certificate83 import certified_lower
            lower=certified_lower(x,obs,self.par,u,self.config["control_dt"],.5)
            if np.min(lower)>=2e-7:break
            if not np.isfinite(lower).all():return None,"candidate_certificate_domain_failure",{}
            # Only uncertified rows require a new search cut. Every candidate
            # is independently recertified for ALL obstacles before acceptance.
            bad=lower<2e-7
            required=np.maximum(0.,B@u+C-tracking-lower)+1e-4
            sampling[bad]=np.maximum(sampling[bad],required[bad])
        else:
            from park_certificate_recovery import recover
            u,record=recover(self,x,obs,B,C,tracking,extraB,extraC)
            self.last_certificate_recovery=record
            if u is None:return None,"candidate_certificate_not_converged",dict(certificate_recovery=record)
        # Recovery changes only the feasibility search, never raw barrier rows.

        self.previous=u.copy();self.last_step_time=float(t);log[:,12]=B@u+C
        off=dict(active=False,j=-1,d=np.zeros(3),gamma=0.,executable_gamma=0.,strength=0.,event=0)
        diag=dict(nominal=nom,base=u,barriers=log,detour=off,compute_seconds=time.perf_counter()-tic,
                  sampling_margin=sampling,tracking_reserve=tracking,certificate_iterations=certificate_iterations,certificate_recovery=getattr(self,"last_certificate_recovery",None))
        return u,"ok",diag
    def advance_auxiliary(self,state,obstacles,dt):
        if dt<=0 or dt>self.config["monitor_dt"]+1e-12:raise ValueError("Invalid step")
        self.time+=float(dt)

    def advance_interval(self,state,obstacles):
        # Baselines keep R/V/reference fixed and do not own an RA last_result.
        # Match their original auxiliary time advance without the RA-only fast path.
        self.time+=float(self.config["control_dt"])
