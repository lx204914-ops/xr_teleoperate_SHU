"""Matched scenes, all-obstacle RA-DPCBF and the supplied executable detour.

The environment owns its tape. The controller receives only the current p,v,a.
Original reference samples and bounded-motion counterparts are different runs.
Every unsuccessful run is retained, with its actual termination reason.
"""
from pathlib import Path
import os, tempfile, json, argparse, time, hashlib
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'ra_reference_20261016'))
import numpy as np
from numba import njit
from scene_tapes import load_scene,state_at,SCENES
from dynamic_study.kernel import constraints,radius_ref,boundary_flow
from nominal_mpc import matrices,solve
from sampled_safety import interval_margins
from safety_filter import SafetyFilter,VARIANTS
from configuration import load_config,config_sha256,output_root
from feasibility_checks import check_initial_state,check_current_sets

ROOT=Path(__file__).resolve().parent

def json_safe(value):
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,(float,np.floating)):return float(value) if np.isfinite(value) else None
    if isinstance(value,np.integer):return int(value)
    if isinstance(value,np.bool_):return bool(value)
    return value

@njit(cache=True)
def transport_step(d,old,e):
    axis=np.cross(old,e);s=np.linalg.norm(axis);c=np.dot(old,e)
    if s>1e-12:
        a=axis/s;d=c*d+s*np.cross(a,d)+(1-c)*a*np.dot(a,d)
    d=d-e*np.dot(e,d)
    return d/max(np.linalg.norm(d),1e-15)

@njit(cache=True)
def advance(x,R,V,ref,u,par,method,tape,td,mode,t,dt,fine,j,d,e,kp,strength):
    minima=np.full(7,1e20);length=0.;maxa=0.;detmin=1e20
    for l in range(int(round(dt/fine))):
        before=x[:3].copy();x[:3]+=fine*x[3:]+.5*fine*fine*u;x[3:]+=fine*u
        length+=np.linalg.norm(x[:3]-before);boundary_flow(R,V,ref,fine,par[2])
        obs=state_at(tape,t+(l+1)*fine,td,mode);B,C,log=constraints(x,obs,R,V,ref,par,method)
        for i in range(len(obs)):
            body=log[i,0]-par[21]-obs[i,9]
            minima[0]=min(minima[0],body);minima[1]=min(minima[1],body-par[22])
            minima[2]=min(minima[2],log[i,6]);minima[3]=min(minima[3],log[i,7])
            minima[4]=min(minima[4],np.dot(B[i],u)+C[i]);minima[6]=min(minima[6],log[i,0])
            maxa=max(maxa,np.linalg.norm(obs[i,6:9]))
        if j>=0:
            r=obs[j,:3]-x[:3];new_e=r/np.linalg.norm(r);d=transport_step(d,e,new_e);e=new_e
            w=obs[j,3:6]-x[3:];q=np.dot(e,w);chi=-np.dot(d,w)
            value=np.dot(d,u-obs[j,6:9])+(kp-q/np.linalg.norm(r))*chi-strength
            detmin=min(detmin,value)
    return x,R,V,d,e,minima,length,maxa,detmin

def run(name,mode='bounded',variant='detour',seed=17,duration=35.,out='results/round1',
        dt=None,nu=None,cruise=None,tr=None,save_history=True):
    config=load_config()
    if variant not in VARIANTS:raise ValueError('Unknown variant: '+variant)
    if not np.isfinite(duration) or duration<=0:raise ValueError('duration must be finite and positive.')
    if config['acceleration_component_limits']!=[2.,2.,2.] or config['obstacle_acceleration_bound']!=.8:
        raise ValueError('This matched-scene backend uses axis bounds ±2 and obstacle bound .8; update both the QP and braking scale before changing them.')
    if nu is not None:config['detour']['nu_req']=nu
    par=np.array(config['parameters'],float)
    if tr is not None:par[0]=tr
    if cruise is not None:par[9]=cruise
    if variant=='no_transverse':
        par[1]=0.;config['gamma_perp']=0.
    if not (np.isfinite(par).all() and par[0]>0 and 0<=par[1]<=1 and par[2]>0 and par[3]>0 and par[5]>0 and par[9]>0 and par[19]>0):
        raise ValueError('Require Tr, omega, kh, span and Zmax positive, and 0<=gamma_perp<=1.')
    dcfg=config['detour']
    if not (dcfg['nu_req']>0 and 0<dcfg['eta']<1 and dcfg['gamma_min']>0 and dcfg['k_perp']>0):
        raise ValueError('Invalid detour strength, usage fraction, capacity threshold or gain.')
    config['parameters']=par.tolist()
    dt=config['control_dt'] if dt is None else dt;fine=min(config['monitor_dt'],dt)
    if not np.isfinite(dt) or dt<=0 or fine<=0 or abs(dt/fine-round(dt/fine))>1e-8:raise ValueError('positive control_dt must be an integer multiple of monitor_dt')
    config['control_dt']=dt;config['monitor_dt']=fine;config['cruise_speed']=float(par[9])
    scene=load_scene(name,mode,seed,duration);tape=scene['tape'];td=scene['tape_dt'];kind=scene['mode']
    requested_duration=duration
    duration=min(duration,(len(tape)-1)*td)
    method=3 if variant=='fixed_radius' else 5
    if variant=='fixed_large':method=7
    if variant=='nominal':method=0
    if variant=='omit_Rdot_diagnostic':method=6
    safety_filter=SafetyFilter(config,variant)
    mechanism=safety_filter.mechanism
    x=scene['start'].copy();obs=state_at(tape,0.,td,kind);n=len(obs)
    R=par[21]+par[22]+obs[:,9];V=np.zeros(n);ref=R.copy();previous=np.zeros(3)
    warm=np.zeros((config['mpc_horizon'],3));mpc=matrices(horizon=config['mpc_horizon'],dt=config['mpc_prediction_dt'],
       position_weight=config['mpc_position_weight'],velocity_weight=config['mpc_velocity_weight'],
       input_weight=config['mpc_input_weight'],increment_weight=config['mpc_increment_weight'])
    histories=[];barriers=[];obstacles=[];detours=[];marginlog=[];zupperlog=[];timings=[]
    auxiliary_diagnostics=[];lifecycle=[];detour_reasons=[];conditional_gate_log=[]
    initial_checks=None;initial_sets=None;final_sets=None
    minima=np.full(7,1e20);path=0.;energy=0.;maxa=0.;maxu=0.;detmin=1e20;status='timeout';wp=0
    active_steps=0;total_steps=0;active_segments=set();epoch_max=0
    auxiliary_binding_steps=0;auxiliary_changed_steps=0
    max_actual_z=-np.inf;max_interval_z=-np.inf;min_sampling_slack=np.inf
    workspace=scene['workspace'];inside_workspace=True
    # Compile kernels on a disposable state before timing any controller update.
    B,C,log=constraints(x,obs,R,V,ref,par,method)
    solve(x,scene['goal'],par[9],previous,warm,*mpc)
    safety_filter.prepare(n)
    interval_margins(log,R,V,ref,par,dt,method)
    advance(x.copy(),R.copy(),V.copy(),ref,previous,par,method,tape,td,kind,0.,dt,fine,-1,
            np.array([0.,1.,0.]),np.array([1.,0.,0.]),config['detour']['k_perp'],0.)
    started=time.perf_counter()
    for k in range(int(round(duration/dt))+1):
        t=k*dt;obs=state_at(tape,t,td,kind)
        for i in range(n):
            r=obs[i,:3]-x[:3];q=r@(obs[i,3:6]-x[3:])/np.linalg.norm(r)
            ref[i],_=radius_ref(q,par[21]+par[22]+obs[i,9],par)
        B,C,log=constraints(x,obs,R,V,ref,par,method)
        goal=scene['waypoints'][wp]
        if np.linalg.norm(x[:3]-goal)<1. and wp<len(scene['waypoints'])-1:
            wp+=1;goal=scene['waypoints'][wp]
        # Common mission evaluation: first positional arrival, before another
        # control/QP request; no group receives a parking-speed requirement.
        mission_distance=float(np.linalg.norm(x[:3]-scene['goal']))
        mission_body=np.linalg.norm(obs[:,:3]-x[:3],axis=1)-par[21]-obs[:,9]
        mission_collision=bool(np.min(mission_body)<=0 or minima[0]<=0)
        mission_arrival=bool(mission_distance<.25 and wp==len(scene['waypoints'])-1)
        mission_initial_reject=bool(k==0 and method and (np.min(log[:,6])<-1e-8 or np.min(log[:,7])<-1e-8))
        mission_timeout=bool(k>=round(duration/dt))
        if mission_collision or mission_initial_reject or mission_arrival or mission_timeout:
            B,C,log=constraints(x,obs,R,V,ref,par,method)
            status='collision' if mission_collision else ('initial_rejected' if mission_initial_reject else ('goal' if mission_arrival else 'timeout'))
            filter_status='mission_terminal_no_control'
            histories.append(np.r_[t,x,np.full(3,np.nan),np.full(3,np.nan),mission_distance,0,np.nan,0,np.nan])
            timings.append([0.,0.,0.,0.,0.])
            minima[:4]=np.minimum(minima[:4],[mission_body.min(),(mission_body-par[22]).min(),log[:,6].min(),log[:,7].min()])
            if save_history:
                barriers.append(log);obstacles.append(obs);marginlog.append(np.zeros(n));zupperlog.append(log[:,2].copy())
                detours.append(np.zeros(13));auxiliary_diagnostics.append([False,False,np.nan,np.nan])
                lifecycle.append([t,0.,False,False,False,False]);detour_reasons.append('mission_terminal_no_control')
                conditional_gate_log.append([np.nan,np.nan,np.nan,np.nan,np.nan,0.,0.,0.,False,0.,False])
            break
        tick=time.perf_counter();nominal,plan,it,mpc_err=solve(x,goal,par[9],previous,warm,*mpc)
        warm[:-1]=plan[1:];warm[-1]=plan[-1];t1=time.perf_counter()
        # Only the bounded-motion family satisfies the declared acceleration/jerk bounds.
        if kind==1 and method!=0:
            margin,zupper,L=interval_margins(log,R,V,ref,par,dt,method)
        else:margin=np.zeros(n);zupper=log[:,2].copy()
        Cexec=C-margin;t2=time.perf_counter()
        if k==0:
            initial_sets=check_current_sets(B,C,config,margin) if method else None
            if variant in ('detour','conditional_detour','no_detour','fixed_direction','fixed_strength','no_transverse'):
                initial_checks=check_initial_state(x,obs,R,V,ref,config)
        if method:
            safety_filter.prediction_plan=plan.copy()
            filtered=safety_filter.solve(x,goal,obs,log,B,C,nominal,previous,dt,margin,certified=kind==1)
            t4=time.perf_counter()
            det=filtered.detour;ok=filtered.control is not None;qp_iters=filtered.qp_iterations
            u=filtered.control if ok else np.full(3,np.nan)
            direction_seconds=filtered.direction_seconds;qp_seconds=filtered.qp_seconds
            filter_status=filtered.status
            auxiliary_binding_steps+=int(filtered.auxiliary_binding)
            auxiliary_changed_steps+=int(filtered.auxiliary_changed_input)
            if save_history:
                auxiliary_diagnostics.append([filtered.auxiliary_binding,filtered.auxiliary_changed_input,
                    filtered.minimum_execution_residual if ok else np.nan,
                    filtered.minimum_augmented_residual if filtered.minimum_augmented_residual is not None else np.nan])
                conditional_gate_log.append([det.get('minimum_common_set_margin_mps2',np.nan),
                    det.get('minimum_individual_row_reserve_mps2',np.nan),
                    det.get('radial_brake_mps2',np.nan),det.get('transverse_correction_mps2',np.nan),
                    det.get('window_progress_m',np.nan),det.get('feasibility_elapsed_s',0.),
                    det.get('radial_braking_elapsed_s',0.),det.get('stagnation_elapsed_s',0.),
                    det.get('safe_progress',False),det.get('release_elapsed_s',0.),det.get('rearmed',False)])
        else:
            u=nominal;ok=True;qp_iters=0;direction_seconds=0.;qp_seconds=0.;filter_status='nominal'
            t4=time.perf_counter()
            det=dict(active=False,j=-1,d=np.zeros(3),chi=0.,gamma=0.,strength=0.,hold_margin=0.,
                     lower=0.,segment=0,event=0,capacity_lp_calls=0)
            if save_history:auxiliary_diagnostics.append([False,False,np.nan,np.nan])
            if save_history:conditional_gate_log.append([np.nan,np.nan,np.nan,np.nan,np.nan,0.,0.,0.,False,0.,False])
        res=B@u+C if ok else np.full(n,np.nan);log[:,12]=res
        dist=np.linalg.norm(x[:3]-scene['goal'])
        histories.append(np.r_[t,x,u if ok else np.full(3,np.nan),nominal,dist,qp_iters,np.linalg.norm(u-nominal),it,mpc_err])
        if save_history:
            barriers.append(log);obstacles.append(obs);marginlog.append(margin);zupperlog.append(zupper)
            detours.append(np.r_[det['active'],det['j'],det['d'],det['chi'],det['gamma'],det.get('executable_gamma',0.),
                               det['strength'],det['hold_margin'],det['segment'],det['event'],det['lower']])
            lifecycle.append([t,det.get('off_elapsed',0.),det.get('primary_clear',False),det.get('primary_receding',False),
                              det.get('goal_clear',False),det.get('nominal_admissible',False)])
            detour_reasons.append(det.get('reason','nominal'))
        total_steps+=1;active_steps+=int(det['active']);epoch_max=max(epoch_max,int(det['segment']))
        if det['active']:active_segments.add(int(det['segment']))
        max_actual_z=max(max_actual_z,float(np.max(log[:,2])));max_interval_z=max(max_interval_z,float(np.max(zupper)))
        if ok and method:min_sampling_slack=min(min_sampling_slack,float(np.min(log[:,12]-margin)))
        inside_workspace=inside_workspace and bool(np.all((x[:3]>=workspace[:,0]+par[21])&(x[:3]<=workspace[:,1]-par[21])))
        # Includes initial certificate work when present; it is not hidden in
        # a real-time claim. Direction and both QP solves are timed separately.
        timings.append([t1-tick,t2-t1,direction_seconds,qp_seconds,t4-tick])
        body=log[:,0]-par[21]-obs[:,9]
        minima[:4]=np.minimum(minima[:4],[body.min(),(body-par[22]).min(),log[:,6].min(),log[:,7].min()])
        minima[5]=min(minima[5],float(np.nanmin(res)) if ok else 1e20)
        if k==0 and method and (minima[2]<-1e-8 or minima[3]<-1e-8):status='initial_rejected';break
        if not ok:
            status='qp_infeasible' if filter_status=='base_qp_infeasible' else filter_status
            final_sets=check_current_sets(B,C,config,margin)
            break
        if body.min()<=0:status='collision';break
        if dist<.25 and wp==len(scene['waypoints'])-1:status='goal';break
        if k>=round(duration/dt):break
        maxu=max(maxu,float(np.max(np.abs(u))));energy+=dt*np.sum((u-nominal)**2)
        j=det['j'] if det['active'] else -1
        d=det['d'];e=mechanism.e
        x,R,V,d,e,mins,length,acc,dmin=advance(x,R,V,ref,u,par,method,tape,td,kind,t,dt,fine,
                                             j,d,e,config['detour']['k_perp'],det['strength'])
        if j>=0:mechanism.d=d;mechanism.e=e
        minima=np.minimum(minima,mins);path+=length;maxa=max(maxa,acc);detmin=min(detmin,dmin);previous=u.copy()
    hist=np.array(histories);times=np.array(timings)
    if save_history:
        logs=np.array(barriers);obs=np.array(obstacles);det=np.array(detours)
        margin=np.array(marginlog);zupper=np.array(zupperlog)
    dest=output_root(out)/mode/name/variant;dest.mkdir(parents=True,exist_ok=True)
    if save_history:
        ids=np.unique(np.r_[np.arange(0,len(hist),20),len(hist)-1]).astype(int)
        np.savez_compressed(dest/'history.npz',history=hist,barriers=logs[ids],
            barrier_indices=ids,parameters=par,tape_dt=td,motion_mode=kind,
            workspace=scene['workspace'],initial=scene['start'],goal=scene['goal'],
            waypoints=scene['waypoints'],detour=det,controller_timing=times,
            auxiliary_diagnostics=np.asarray(auxiliary_diagnostics),
            detour_lifecycle=np.asarray(lifecycle),detour_reason=np.asarray(detour_reasons),
            conditional_gate=np.asarray(conditional_gate_log),
            sampling_margins=margin[ids],interval_Z_upper=zupper[ids])
    row=dict(scene=name,mode=mode,variant=variant,seed=seed,
       manoeuvre_seed=scene.get('manoeuvre_seed',seed if kind==1 else None),
       geometry_seed_source=scene.get('geometry_seed_source','reference_inputs/manifest.json; the manoeuvre seed does not change the frozen reference layout'),
       obstacles=n,termination=status,reached_goal=status=='goal',
       duration_s=float(hist[-1,0]),final_error_m=float(hist[-1,13]),collision=bool(minima[0]<=0),
       qp_failures=int(status in ('qp_infeasible','sampling_set_infeasible','base_qp_numerical_failure',
           'augmented_qp_numerical_failure','execution_qp_numerical_failure','input_box_numerical_failure',
           'feasibility_numerical_uncertainty')),
       min_body_clearance_m=float(minima[0]),min_base_margin_m=float(minima[1]),min_dynamic_clearance_m=float(minima[2]),
       min_h=float(minima[3]),min_intersample_CBF_residual=float(minima[4]) if len(hist)>1 else None,
       min_control_CBF_residual=float(minima[5]) if minima[5]<1e19 else None,
       min_sampling_certificate_slack=float(min_sampling_slack) if method and status!='qp_infeasible' and np.isfinite(min_sampling_slack) else None,
       min_detour_intersample_residual=float(detmin) if active_steps>0 and detmin<1e19 else None,
       detour_active_fraction=float(active_steps/max(total_steps,1)),detour_segments=int(len(active_segments)),detour_epoch_counter=int(epoch_max),
       max_obstacle_acceleration_norm=float(maxa),declared_acceleration_bound=.8,declared_jerk_bound=1.1,
       sampled_data_certificate_applied=kind==1 and method!=0,max_acceleration_component=float(maxu),path_length_m=float(path),
       correction_integral=float(energy),max_actual_Z=float(max_actual_z),max_interval_Z_bound=float(max_interval_z),
       robot_inside_reference_workspace=bool(inside_workspace),
       obstacle_tape_sha256=hashlib.sha256(tape.tobytes()).hexdigest(),same_reference_knots=scene['same_reference_knots'],
       config=config,executed_parameters=par.tolist(),requested_horizon_s=requested_duration,environment_horizon_s=(len(tape)-1)*td,
       control_dt=dt,monitor_dt=fine,runtime_s=time.perf_counter()-started,
       timing_ms_p50=np.quantile(times,[.5],axis=0).ravel().tolist(),timing_ms_p95=np.quantile(times,[.95],axis=0).ravel().tolist(),
       timing_ms_max=np.max(times,axis=0).ravel().tolist(),history_saved=bool(save_history),horizon_exhausted=status=='timeout',
       timing_order=['MPC','sample_margin','direction_and_capacity','QP','total'],
       method_config_sha256=config_sha256(config),filter_termination_status=filter_status,
       qp_independent_solver_fallbacks=int((safety_filter.base_qp.fallbacks if safety_filter.base_qp else 0)+
          (safety_filter.augmented_qp.fallbacks if safety_filter.augmented_qp else 0)),
       hard_qp_matrix_nnz=int(safety_filter.base_qp.nnz if safety_filter.base_qp else 0),
        main_method_variant=variant in ('detour','conditional_detour'),
        conditional_detour_copy=variant=='conditional_detour',
       auxiliary_binding_steps=int(auxiliary_binding_steps),
       auxiliary_changed_control_steps=int(auxiliary_changed_steps),
        conditional_gate_summary=(safety_filter.conditional_gate.summary() if safety_filter.conditional_gate is not None else None),
        initial_feasibility_checks=initial_checks,initial_control_sets=initial_sets,terminal_control_sets=final_sets,
       execution_scope=('Continuous bounded-motion counterpart' if kind==1 else
         'Original discrete p/v/a records with native reflection/contact resets; no global bounded-acceleration certificate' if kind==2 else
         'C2 interpolation audit only: interpolation pulses invalidate the declared acceleration bound'))
    row['timing_ms_p50']=[v*1000 for v in row['timing_ms_p50']];row['timing_ms_p95']=[v*1000 for v in row['timing_ms_p95']]
    row['timing_ms_max']=[v*1000 for v in row['timing_ms_max']]
    row=json_safe(row)
    (dest/'metrics.json').write_text(json.dumps(row,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps({k:row[k] for k in ('scene','mode','variant','termination','duration_s','min_body_clearance_m','min_h','detour_segments','runtime_s')},ensure_ascii=False),flush=True)
    return row

def main():
    p=argparse.ArgumentParser(description='Current RA_DPCBF_1: unchanged barrier, executable detour, explicit feasibility checks.')
    p.add_argument('--scenes',nargs='+',required=True);p.add_argument('--modes',nargs='+',default=['bounded'],choices=['reference','bounded'])
    p.add_argument('--variants',nargs='+',default=['detour'],choices=VARIANTS);p.add_argument('--seed',type=int,default=17)
    p.add_argument('--duration',type=float,default=35.);p.add_argument('--out',default='results/round1');p.add_argument('--dt',type=float)
    p.add_argument('--nu',type=float);p.add_argument('--cruise',type=float);p.add_argument('--tr',type=float)
    a=p.parse_args();rows=[]
    for mode in a.modes:
        for scene in a.scenes:
            for variant in a.variants:rows.append(run(scene,mode,variant,a.seed,a.duration,a.out,a.dt,a.nu,a.cruise,a.tr))
    dest=output_root(a.out);dest.mkdir(parents=True,exist_ok=True)
    (dest/'last_batch.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':main()
