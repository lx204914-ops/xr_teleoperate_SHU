"""Stateful adapter for the frozen controller; no safety or selection equations changed."""
from pathlib import Path
import copy, json, os, sys, time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
for name in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMBA_NUM_THREADS"):
    os.environ[name]="1"
os.environ["NUMBA_CACHE_DIR"]=str(ROOT/"cache")
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/"frozen_controller"))
import numpy as np
from configuration import load_config
from dynamic_study.kernel import radius_ref,boundary_flow,constraints
from nominal_mpc import matrices,solve
from sampled_safety import interval_margins
from directional_interval import interval_margin_rows
from deployment_safety_filter import SafetyFilter
from deployment_stable_progress import StableProgressDetour
from numba import njit
@njit(cache=True)
def transport_step(d,old,e):
    axis=np.cross(old,e);s=np.linalg.norm(axis);c=np.dot(old,e)
    if s>1e-12:
        a=axis/s;d=c*d+s*np.cross(a,d)+(1-c)*a*np.dot(a,d)
    d=d-e*np.dot(e,d)
    return d/max(np.linalg.norm(d),1e-15)
class DeployedController:
    def solve_nominal(self,*args):
        return solve(*args,limit=self.mpc_input_limit)
    def __init__(self):
        self.config=load_config(ROOT/"frozen_controller/method_config.json")
        self.config["experiment_selector"]="progress_with_tick_hysteresis"
        self.config["control_dt"]=0.005  # deployment period; original source retained
        # Derived once from the actual execution model, never from scene results.
        from rotorpy.vehicles.crazyflie_params import quad_params
        self.execution_time_scale=max(quad_params["tau_m"]/4.0,1.0/np.sqrt(quad_params["kp_att"]))
        self.config["execution_smoothing"]=self.execution_time_scale/self.config["control_dt"]
        self.ready=False
    def reset(self,state,goal,obstacles,obstacle_ids=None):
        x=np.asarray(state,dtype=float);obs=np.asarray(obstacles,dtype=float)
        if x.shape!=(6,) or obs.ndim!=2 or obs.shape[1]!=10:raise ValueError("Expected x[6], obstacles[N,10]")
        if not np.isfinite(x).all() or not np.isfinite(obs).all():raise ValueError("Nonfinite initial state")
        self.goal=np.asarray(goal,dtype=float).copy()
        if self.goal.shape!=(3,) or not np.isfinite(self.goal).all():raise ValueError("Invalid goal")
        self.ids=tuple(range(len(obs))) if obstacle_ids is None else tuple(obstacle_ids)
        if len(self.ids)!=len(obs) or len(set(self.ids))!=len(obs):raise ValueError("Stable unique obstacle IDs required")
        cfg=self.config;self.par=np.asarray(cfg["parameters"],dtype=float)
        self.R=self.par[21]+self.par[22]+obs[:,9];self.V=np.zeros(len(obs));self.ref=self.R.copy()
        self.previous=np.zeros(3);self.warm=np.zeros((cfg["mpc_horizon"],3))
        self.mpc=matrices(horizon=cfg["mpc_horizon"],dt=cfg["mpc_prediction_dt"],
            position_weight=cfg["mpc_position_weight"],velocity_weight=cfg["mpc_velocity_weight"],
            input_weight=cfg["mpc_input_weight"],increment_weight=cfg["mpc_increment_weight"])
        self.filter=SafetyFilter(cfg,"conditional_detour")
        self.filter.mechanism=StableProgressDetour(cfg["detour"],cfg)
        self.filter.prepare(len(obs)+6);self.time=0.;self.last_step_time=None;self.last_result=None
        # Disposable warm-up does not advance controller or boundary state.
        self.solve_nominal(x,self.goal,self.par[9],self.previous,self.warm,*self.mpc)
        self.execution_reference=np.zeros(3);self.execution_reference_rate=np.zeros(3)
        self.ready=True
    def initialize_execution(self,command):
        self.previous=np.asarray(command,dtype=float).copy()
        self.execution_reference=self.previous.copy();self.execution_reference_rate=np.zeros(3)
    def advance_reference(self,nominal):
        target=np.clip(nominal,-1.5,1.5);omega=1./self.execution_time_scale;dt=self.config["control_dt"]
        y=self.execution_reference-target;temp=self.execution_reference_rate+omega*y
        decay=np.exp(-omega*dt)
        self.execution_reference=target+(y+temp*dt)*decay
        self.execution_reference_rate=(self.execution_reference_rate-omega*temp*dt)*decay
        return self.execution_reference.copy()
    def step(self,t,state,obstacles,obstacle_ids=None):
        if not self.ready:raise RuntimeError("Call reset first")
        if abs(float(t)-self.time)>1e-8:raise ValueError("Auxiliary states must be advanced with physical time")
        if self.last_step_time is not None and abs(t-self.last_step_time-self.config["control_dt"])>1e-8:
            raise ValueError("Unexpected control interval")
        x=np.asarray(state,dtype=float);obs=np.asarray(obstacles,dtype=float)
        ids=self.ids if obstacle_ids is None else tuple(obstacle_ids)
        if x.shape!=(6,) or obs.shape!=(len(self.ids),10) or ids!=self.ids:raise ValueError("State/obstacle identity mismatch")
        if not np.isfinite(x).all() or not np.isfinite(obs).all():raise ValueError("Nonfinite input")
        tic=time.perf_counter()
        for i in range(len(obs)):
            r=obs[i,:3]-x[:3];D=np.linalg.norm(r)
            if D<=1e-9:raise ValueError("Undefined line of sight")
            q=r@(obs[i,3:6]-x[3:])/D
            self.ref[i],_=radius_ref(q,self.par[21]+self.par[22]+obs[i,9],self.par)
        B,C,log=constraints(x,obs,self.R,self.V,self.ref,self.par,5)
        if self.last_step_time is None and (np.min(log[:,6])<-1e-8 or np.min(log[:,7])<-1e-8):
            return None,"initial_safety_condition_failed",dict(s=log[:,6],h=log[:,7])
        nominal,plan,iterations,error=self.solve_nominal(x,self.goal,self.par[9],self.previous,self.warm,*self.mpc)
        self.warm[:-1]=plan[1:];self.warm[-1]=plan[-1]
        margin,zupper,_,self.interval_refined_rows=interval_margin_rows(x,obs,self.R,self.V,self.ref,self.par,B,C,self.config["control_dt"])
        self.config["execution_reference"]=self.advance_reference(nominal)
        tracking_reserve=0.5*np.linalg.norm(B,axis=1)
        margin=margin+tracking_reserve
        # Only retained physical authority rows; no ad-hoc jerk or
        # measured-acceleration proximity rows are imposed.
        from ground_rows import rows as physical_rows
        self.filter.execution_rows=physical_rows(self,x)
        self.filter.actual_acceleration=np.asarray(self.execution_acceleration).copy()
        self.filter.prediction_plan=plan.copy()
        result=self.filter.solve(x,self.goal,obs,log,B,C,nominal,self.previous,self.config["control_dt"],margin,certified=True)
        self.last_step_time=float(t);self.last_result=result
        u=result.control
        if u is not None:self.previous=u.copy();log[:,12]=B@u+C
        diag=dict(R=self.R.copy(),V=self.V.copy(),Rref=self.ref.copy(),B=B.copy(),C=C.copy(),
                  barriers=log.copy(),nominal=nominal.copy(),base=result.base_witness,detour=copy.deepcopy(result.detour),
                  sampling_margin=margin.copy(),interval_Z_upper=zupper,mpc_iterations=iterations,mpc_residual=error,
                  compute_seconds=time.perf_counter()-tic,physical_tracking_error_covered="conditional on verified 0.5 norm error and sampled bounds",tracking_reserve=tracking_reserve)
        return None if u is None else u.copy(),result.status,diag
    def advance_auxiliary(self,state,obstacles,dt):
        if self.last_result is None or self.last_result.control is None:raise RuntimeError("No executable current command")
        if dt<=0 or dt>self.config["monitor_dt"]+1e-12:raise ValueError("Invalid auxiliary substep")
        obs=np.asarray(obstacles,dtype=float);x=np.asarray(state,dtype=float)
        if obs.shape!=(len(self.ids),10) or x.shape!=(6,):raise ValueError("Substep state shape")
        boundary_flow(self.R,self.V,self.ref,float(dt),self.par[2])
        if self.last_result.detour["active"]:
            mechanism=self.filter.mechanism;j=mechanism.j
            if j>=0:
                r=obs[j,:3]-x[:3];e=r/np.linalg.norm(r)
                mechanism.d=transport_step(mechanism.d,mechanism.e,e);mechanism.e=e.copy()
        self.time+=float(dt)
