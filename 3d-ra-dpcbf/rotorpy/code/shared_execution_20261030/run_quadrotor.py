"""Six-DoF closure of the unchanged RA controller on an existing frozen scene."""
from pathlib import Path
import argparse,hashlib,json,time,copy,sys
ROOT=Path(__file__).resolve().parents[1]

sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(ROOT))
from deployed_controller import DeployedController as Controller
from baseline_adapter import BaselineController
from common_initialization import common_trim
import numpy as np
from rotorpy.vehicles.multirotor import Multirotor

from executor import AccelerationExecutor, CONFIG
from rotorpy.vehicles.crazyflie_params import quad_params
from dynamic_study.kernel import constraints
def obstacle_state(tape,td,t):
    if t>=(len(tape)-1)*td-1e-12:return tape[-1].copy()
    k=min(int(np.floor((t+1e-10)/td)),len(tape)-2);tau=max(0.,min(1.,(t-k*td)/td))
    ds=tau*td;a=tape[k];b=tape[k+1];j=(b[:,6:9]-a[:,6:9])/td;o=a.copy()
    o[:,:3]=a[:,:3]+ds*a[:,3:6]+.5*ds**2*a[:,6:9]+ds**3/6*j
    o[:,3:6]=a[:,3:6]+ds*a[:,6:9]+.5*ds**2*j;o[:,6:9]=a[:,6:9]+ds*j
    return o
def run(scene_path,horizon=24.,method="ra"):
    scene_path=Path(scene_path)
    digest=hashlib.sha256(scene_path.read_bytes()).hexdigest()
    if scene_path.name=="four_obstacle_scene.npz" and digest!="45c591518e0f1e3d1fe3d176a16cc6b4cf1c26d895a80ee60248d189afac5b7e":
        raise ValueError("Frozen four-obstacle scene changed")
    with np.load(scene_path,allow_pickle=False) as z:
        tape=z["tape"];td=float(z["tape_dt"]);assert int(z["mode"])==1
        start=z["start"];goal=z["goal"];workspace=z["workspace"]
    case_name="four_obstacle" if scene_path.name=="four_obstacle_scene.npz" else scene_path.stem
    tag="prioritized_execution"
    out=Path(__file__).resolve().parent/"results"/f"{case_name}_{method}_{tag}"
    if out.exists():raise FileExistsError("Prior results preserved: "+str(out))
    out.mkdir(parents=True)
    obs=obstacle_state(tape,td,0)
    trim=common_trim(start,goal,obs)
    # Disposable initialization keeps compilation/setup outside mission timing.
    warm=Controller() if method=="ra" else BaselineController(method)
    warm.reset(start,goal,obs);warm.initialize_execution(trim["acceleration"])
    warm.execution_acceleration=np.asarray(trim["acceleration"])
    warm.step(0.,start,obs)
    controller=Controller() if method=="ra" else BaselineController(method)
    controller.reset(start,goal,obs);controller.initialize_execution(trim["acceleration"])
    dt=controller.config["control_dt"];fine=controller.config["monitor_dt"];steps=round(dt/fine)
    horizon=min(float(horizon),(len(tape)-1)*td)
    params=copy.deepcopy(quad_params)
    hover=np.sqrt(params["mass"]*9.81/(params["num_rotors"]*params["k_eta"]))
    state=dict(x=start[:3].copy(),v=start[3:].copy(),q=np.asarray(trim["quaternion_xyzw"]),w=np.zeros(3),
               wind=np.zeros(3),rotor_speeds=np.asarray(trim["rotor_speeds"]))
    vehicle=Multirotor(params,initial_state=state,control_abstraction="cmd_motor_speeds",aero=True)

    executor=AccelerationExecutor(vehicle)
    rows=[];obs_rows=[];barriers=[];boundary=[];events=[];timings=[];inner_timings=[];angular_rates=[]

    min_h=np.inf;min_s=np.inf;negative_actual_steps=0;max_actual_axis=0.;max_command_jerk=0.;previous_u=np.asarray(trim["acceleration"]).copy();bound_violation_steps=0;min_detour=np.inf;active_detour_substeps=0
    arrival=None;mission="timeout";post="not_started";min_gap=float(np.min(np.linalg.norm(obs[:,:3]-start[:3],axis=1)-controller.par[21]-obs[:,9]));post_gap=np.inf
    min_cmd=np.inf;min_actual=np.inf;max_error=0.;error_squares=0.;samples=0;saturated=0;path=0.
    previous_position=state["x"].copy();start_wall=time.perf_counter();failure=None
    for k in range(round(horizon/dt)+1):
        t=k*dt;obs=obstacle_state(tape,td,t);x=np.r_[state["x"],state["v"]]
        distance=float(np.linalg.norm(x[:3]-goal))
        if arrival is None and distance<.25:arrival=t;mission="goal";post="running"
        if t>=horizon-1e-12 or (arrival is not None and t>=arrival+2.-1e-12):
            if arrival is not None:post="completed" if t>=arrival+2.-1e-12 else "limited_by_tape"
            rows.append(np.r_[t,x,np.full(12,np.nan),state["q"],state["rotor_speeds"],distance,arrival is not None])
            break
        controller.execution_acceleration=vehicle.statedot(
            state,{"cmd_motor_speeds":state["rotor_speeds"]},fine)["vdot"]
        u,status,diag=controller.step(t,x,obs)
        if u is None:
            failure=dict(time_s=t,status=status)
            rows.append(np.r_[t,x,np.full(12,np.nan),state["q"],state["rotor_speeds"],distance,arrival is not None])
            if arrival is None:mission=status
            else:post=status
            break
        max_command_jerk=max(max_command_jerk,float(np.max(np.abs(u-previous_u)))/dt)
        previous_u=u.copy()
        inner_elapsed=0.;inner_tick=time.perf_counter()
        control=executor.command(state,u,fine,new_reference=True,reference_dt=dt)
        inner_elapsed+=time.perf_counter()-inner_tick
        a0=vehicle.statedot(state,control,fine)["vdot"]
        rows.append(np.r_[t,x,u,diag["nominal"],a0,diag["base"],state["q"],state["rotor_speeds"],distance,arrival is not None])
        angular_rates.append(state["w"].copy())

        obs_rows.append(obs);barriers.append(diag["barriers"]);boundary.append(np.c_[controller.R,controller.V,controller.ref])
        d=diag["detour"];events.append([t,d["active"],d["j"],*d["d"],d["gamma"],d.get("executable_gamma",0.),d["strength"],d["event"]])
        timings.append(diag["compute_seconds"])
        stopped=False
        for sub in range(steps):
            if sub>0:
                inner_tick=time.perf_counter()
                control=executor.command(state,u,fine,new_reference=False)
                inner_elapsed+=time.perf_counter()-inner_tick

            state=vehicle.step(state,control,fine)
            if not all(np.isfinite(state[key]).all() for key in ("x","v","q","w","rotor_speeds")):
                failure=dict(time_s=t+(sub+1)*fine,status="nonfinite_physical_state");stopped=True;break
            sub_t=t+(sub+1)*fine;current_obs=obstacle_state(tape,td,sub_t);current_x=np.r_[state["x"],state["v"]]
            controller.advance_auxiliary(current_x,current_obs,fine)
            actual=vehicle.statedot(state,control,fine)["vdot"]
            if method=="ra":B,C,checklog=constraints(current_x,current_obs,controller.R,controller.V,controller.ref,controller.par,5)
            else:B,C,checklog=controller.raw_coefficients(current_x,current_obs)
            cmd_res=float(np.min(B@u+C));actual_res=float(np.min(B@actual+C))
            if method=="ra" and d["active"]:
                j=int(d["j"]);direction=controller.filter.mechanism.d
                relative=current_obs[j,:3]-state["x"];distance_j=np.linalg.norm(relative);e=relative/distance_j
                relative_v=current_obs[j,3:6]-state["v"];q=e@relative_v;z=relative_v-q*e
                chi=-direction@z;kp=controller.config["detour"]["k_perp"]
                actual_detour=direction@(actual-current_obs[j,6:9])+(kp-q/distance_j)*chi-d["strength"]
                min_detour=min(min_detour,float(actual_detour));active_detour_substeps+=1
            gap=float(np.min(np.linalg.norm(current_obs[:,:3]-state["x"],axis=1)-controller.par[21]-current_obs[:,9]))
            if arrival is None:min_gap=min(min_gap,gap)
            else:post_gap=min(post_gap,gap)
            min_cmd=min(min_cmd,cmd_res);min_actual=min(min_actual,actual_res)

            min_h=min(min_h,float(np.min(checklog[:,7])));min_s=min(min_s,float(np.min(checklog[:,6])))

            negative_actual_steps+=int(actual_res < -1e-8);max_actual_axis=max(max_actual_axis,float(np.max(np.abs(actual))))
            error=float(np.linalg.norm(actual-u));max_error=max(max_error,error);error_squares+=error**2;samples+=1;bound_violation_steps+=int(error>0.5+1e-8)
            saturated+=int(np.any(state["rotor_speeds"]<=params["rotor_speed_min"]+1e-7) or np.any(state["rotor_speeds"]>=params["rotor_speed_max"]-1e-7))
            if arrival is None:path+=float(np.linalg.norm(state["x"]-previous_position))
            previous_position=state["x"].copy()
            if gap<=0:
                failure=dict(time_s=sub_t,status="collision");stopped=True;break
        inner_timings.append(inner_elapsed)
        if stopped:
            if arrival is None:mission=failure["status"]
            else:post=failure["status"]
            rows.append(np.r_[failure["time_s"],state["x"],state["v"],np.full(12,np.nan),state["q"],state["rotor_speeds"],np.linalg.norm(state["x"]-goal),arrival is not None])
            break
        if k%400==0:print(json.dumps(dict(t_s=t,goal_distance_m=distance,min_body_gap_m=min_gap,mission=mission)),flush=True)
    np.savez_compressed(out/"history.npz",history=np.asarray(rows),obstacles=np.asarray(obs_rows),barriers=np.asarray(barriers),
        boundary=np.asarray(boundary),detour=np.asarray(events),controller_seconds=np.asarray(timings),inner_seconds=np.asarray(inner_timings),
        goal=goal,initial=start,workspace=workspace,angular_rates=np.asarray(angular_rates),scene_sha256=np.array(digest),
        columns=np.array(["t","px","py","pz","vx","vy","vz","ux","uy","uz","unom_x","unom_y","unom_z","actual_ax","actual_ay","actual_az",
                          "ub_x","ub_y","ub_z","qx","qy","qz","qw","motor1","motor2","motor3","motor4","goal_distance","post_arrival"]))
    result=dict(method=method,mission=mission,first_arrival_s=arrival,path_length_to_arrival_m=path,min_body_gap_before_arrival_m=min_gap,
        post_arrival_status=post,post_arrival_min_body_gap_m=post_gap if np.isfinite(post_gap) else None,
        final_position_m=state["x"].tolist(),final_velocity_mps=state["v"].tolist(),final_goal_distance_m=float(np.linalg.norm(state["x"]-goal)),
        min_command_CBF_residual=min_cmd if np.isfinite(min_cmd) else None,min_actual_CBF_residual=min_actual if np.isfinite(min_actual) else None,
        acceleration_tracking_rmse_mps2=float(np.sqrt(error_squares/max(samples,1))),acceleration_tracking_max_norm_mps2=max_error,
        motor_saturated_substeps=saturated,motor_command_clipped_steps=executor.clipped_steps,

        min_h=min_h,min_s=min_s,negative_actual_seconds=negative_actual_steps*fine,max_actual_axis_acceleration=max_actual_axis,

        min_actual_detour_residual=min_detour if np.isfinite(min_detour) else None,active_detour_seconds=active_detour_substeps*fine,execution_preference_counts=(getattr(controller.filter,"preference_counts",{}) if method=="ra" else controller.preference_counts),execution_config=CONFIG,common_initialization=trim,control_dt_s=dt,tracking_bound_violation_steps=bound_violation_steps,max_command_jerk_axis_m_s3=max_command_jerk,control_p95_ms=1000*float(np.quantile(timings,.95)) if timings else None,
        runtime_s=time.perf_counter()-start_wall,scene_sha256=digest,plant="RotorPy Crazyflie six-DoF scalar CPU; shared pre-QP reference governor, common trim, 200Hz",
        execution_smoothing=controller.config["execution_smoothing"],
        total_control_p95_ms=1000*float(np.quantile(np.asarray(timings)+np.asarray(inner_timings),.95)) if timings else None,
        total_control_max_ms=1000*float(np.max(np.asarray(timings)+np.asarray(inner_timings))) if timings else None,
        deadline_overrun_steps=int(np.sum(np.asarray(timings)+np.asarray(inner_timings)>dt)) if timings else 0,failure=failure,
        caveat="Deployment tracking requirement is checked, not globally proved; sampled conditions are independently monitored at 0.5ms.")
    for key,value in list(result.items()):
        if isinstance(value,float) and not np.isfinite(value):result[key]=None
    (out/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    print(json.dumps(result),flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--method",choices=["ra","ma_cbf_vo","c3bf","park"],default="ra");parser.add_argument("--scene",default=str(ROOT/"four_obstacle_scene.npz"));args=parser.parse_args()
    run(args.scene,method=args.method)
