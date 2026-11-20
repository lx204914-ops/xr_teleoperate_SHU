"""Independent math, physical initialization, and no-obstacle execution checks."""
from pathlib import Path
import sys,json,time,copy,hashlib
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(HERE))
from deployed_controller import DeployedController
from baseline_adapter import BaselineController
from candidate_hold_fast import candidate_lower
from common_initialization import common_trim
from executor import AccelerationExecutor
from run_quadrotor import obstacle_state
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
def state_for(initial,trim=None):
    hover=np.sqrt(quad_params["mass"]*9.81/(4*quad_params["k_eta"]))
    return dict(x=initial[:3].copy(),v=initial[3:].copy(),q=np.array([0.,0.,0.,1.]) if trim is None else np.array(trim["quaternion_xyzw"]),
        w=np.zeros(3),wind=np.zeros(3),rotor_speeds=np.full(4,hover) if trim is None else np.array(trim["rotor_speeds"]))
def main():
    out=HERE/"checks"
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    frozen=json.loads((ROOT/"frozen_controller/source_manifest.json").read_text("utf8"))
    assert all(hashlib.sha256((ROOT/"frozen_controller"/n).read_bytes()).hexdigest()==v["sha256"] for n,v in frozen.items())
    with np.load(ROOT/"four_obstacle_scene.npz") as z:initial=z["start"];goal=z["goal"];tape=z["tape"];td=float(z["tape_dt"])
    obs=tape[0];trim=common_trim(initial,goal,obs);state=state_for(initial,trim)
    plant=Multirotor(copy.deepcopy(quad_params),initial_state=state,control_abstraction="cmd_motor_speeds",aero=True)
    actual=plant.statedot(state,{"cmd_motor_speeds":state["rotor_speeds"]},.0005)["vdot"]
    trim_error=float(np.max(abs(actual-np.array(trim["acceleration"]))));assert trim_error<1e-12
    initial_rows={}
    for name in ("ra","ma_cbf_vo","c3bf","park"):
        c=DeployedController() if name=="ra" else BaselineController(name);c.reset(initial,goal,obs);c.initialize_execution(trim["acceleration"])
        c.execution_acceleration=actual
        u,status,diag=c.step(0.,initial,obs);assert status=="ok",(name,status)
        initial_rows[name]=dict(status=status,control=u.tolist(),initial_tracking_error=float(np.linalg.norm(u-actual)))
    # Audit the optimized interval evaluator against exact native formulas,
    # including acceleration error and obstacle jerk; never use future tape.
    old=ROOT/"execution_governor_20261030/results/four_obstacle_park_soft_execution/history.npz"
    with np.load(old) as z:h=z["history"];x=h[-1,1:7];t=h[-1,0]
    o=obstacle_state(tape,td,float(t));c=BaselineController("park");c.reset(initial,goal,obs)
    u=np.zeros(3);dt=.005;lower=candidate_lower(x,o,c.par,u,dt,.5,4)
    assert np.isfinite(lower).all()
    directions=np.r_[np.zeros((1,3)),np.eye(3),-np.eye(3),np.ones((1,3))/np.sqrt(3),-np.ones((1,3))/np.sqrt(3)]
    minimum=np.inf;count=0
    for tau in np.linspace(0,dt,13):
        for e in directions*.5:
            for jerk in directions*1.1:
                xx=x.copy();oo=o.copy();xx[:3]+=tau*x[3:]+.5*tau*tau*(u+e);xx[3:]+=tau*(u+e)
                oo[:,:3]+=tau*o[:,3:6]+.5*tau*tau*o[:,6:9]+tau**3/6*jerk
                oo[:,3:6]+=tau*o[:,6:9]+.5*tau*tau*jerk;oo[:,6:9]+=tau*jerk
                B,C,_=c.raw_coefficients(xx,oo);minimum=min(minimum,float(np.min(B@(u+e)+C-lower)));count+=len(o)
    assert minimum>=-1e-7
    bench=[]
    for k in range(100):
        tic=time.perf_counter();candidate_lower(x,o,c.par,u,dt,.5,4);bench.append(time.perf_counter()-tic)
    checks=[]
    for name in ("step_start","fast_smooth","held_changes"):
        c=DeployedController();c.execution_reference=np.zeros(3);c.execution_reference_rate=np.zeros(3);prev=np.zeros(3)
        state=state_for(np.array([0.,0.,5.,0.,0.,0.]));plant=Multirotor(copy.deepcopy(quad_params),initial_state=state,control_abstraction="cmd_motor_speeds",aero=True)
        exe=AccelerationExecutor(plant);errs=[];rmin=np.inf;rmax=-np.inf;axis=0.;rows=[]
        dt=c.config["control_dt"];fine=.0005;steps=round(dt/fine);S=c.config["execution_smoothing"];s=c.par[6] if hasattr(c,"par") else c.config["parameters"][6]
        for k in range(12000):
            if k%steps==0:
                t=k*fine
                if name=="step_start":nom=np.array([2.,0.,0.]) if t<2. else np.zeros(3)
                elif name=="fast_smooth":nom=1.5*np.sin(t*np.array([4.,3.5,3.]))
                else:nom=np.asarray([[0,0,0],[.8,0,0],[0,.8,0],[-.8,0,.3],[0,-.8,-.3],[0,0,0.]])[int(t//.75)%6]
                ref=c.advance_reference(nom)
                u=np.clip((nom+s*prev+S*ref)/(1+s+S),-1.5,1.5);prev=u.copy()
            ctrl=exe.command(state,u,fine,new_reference=k%steps==0,reference_dt=dt)
            a=plant.statedot(state,ctrl,fine)["vdot"]
            errs.append(float(np.linalg.norm(a-u)));axis=max(axis,float(np.max(abs(a))))
            if k%steps==0:rows.append(np.r_[k*fine,nom,u,a])
            state=plant.step(state,ctrl,fine);rmin=min(rmin,float(np.min(state["rotor_speeds"])));rmax=max(rmax,float(np.max(state["rotor_speeds"])))
        r=dict(test=name,peak_error=max(errs),rmse=float(np.sqrt(np.mean(np.square(errs)))),actual_rotor_min=rmin,actual_rotor_max=rmax,
            max_actual_axis=axis,command_limiting_steps=exe.clipped_steps)
        r["passed"]=bool(max(errs)<=.5 and rmin>=0 and rmax<=2500 and axis<=2.)
        np.savez_compressed(out/(name+".npz"),samples=np.asarray(rows));checks.append(r);print(json.dumps(r),flush=True)
    result=dict(passed=all(r["passed"] for r in checks),frozen_main_files_unchanged=len(frozen),shared_trim=trim,trim_acceleration_error=trim_error,
        initial_QPs=initial_rows,exact_interval_samples_checked=count,interval_inclusion_min_gap=minimum,
        fast_certificate_P95_ms=1000*float(np.quantile(bench,.95)),physical_checks=checks,
        condition="Interval proof assumes tracking error bound over hold; closed-loop runs verify observed errors separately.",
        acceptance="Physical rotor constraints and measured error, command limiting is recorded rather than hidden.")
    (out/"summary.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
