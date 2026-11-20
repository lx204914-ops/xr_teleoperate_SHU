"""Inherited baseline equations, common native MPC/box/integration clock.
No baseline or proposed method is given future obstacle states. A failed QP
projection is checked independently before labeling infeasibility.
"""
from bootstrap import MAIN,LABELS,HERE
from pathlib import Path
import json,time,hashlib
import numpy as np
from numba import njit
from scene_tapes import state_at
from dynamic_study.kernel import project_qp
from nominal_mpc import matrices,solve
from baseline_coefficients import KL,KM
from sampled_baselines import hold_certificates
from ma_cbf_vo import terms,hard_hold,SoftVOQP,monitor_ma

@njit(cache=True)
def monitor_step(x,u,tape,td,t,dt,fine,par,method,branches):
    minimum=np.full(3,1e20);length=0.
    n=len(tape[0]);rad=par[21]+par[22]+tape[0,:,9];zero=np.zeros(n)
    for k in range(round(dt/fine)):
        before=x[:3].copy();x[:3]+=fine*x[3:]+.5*fine*fine*u;x[3:]+=fine*u
        length+=np.linalg.norm(x[:3]-before)
        obs=state_at(tape,t+(k+1)*fine,td,1)
        B,C,log=coefficients(x,obs,rad,zero,rad,par,method)
        if method==10:
            bb,cc,hh=vo_planes(x,obs,par)
            for i in range(n):
                j=branches[i];B[i]=bb[i,j];C[i]=cc[i,j];log[i,7]=hh[i,j]
        for i in range(n):
            body=log[i,0]-par[21]-obs[i,9]
            minimum[0]=min(minimum[0],body)
            minimum[1]=min(minimum[1],log[i,7])
            minimum[2]=min(minimum[2],B[i]@u+C[i])
    return x,minimum,length

def ma_baseline(scene,name,dest,save_history=True):
    cfg=json.loads((MAIN/'method_config.json').read_text(encoding='utf8'))
    par=np.array(cfg['parameters']);dt=cfg['control_dt'];fine=cfg['monitor_dt']
    method=20
    tape=scene['tape'];td=scene['tape_dt'];duration=(len(tape)-1)*td;n=len(scene['tape'][0])
    x=scene['start'].copy();goal=scene['goal'];previous=np.zeros(3)
    warm=np.zeros((cfg['mpc_horizon'],3))
    mpc=matrices(horizon=cfg['mpc_horizon'],dt=cfg['mpc_prediction_dt'],
        position_weight=cfg['mpc_position_weight'],velocity_weight=cfg['mpc_velocity_weight'],
        input_weight=cfg['mpc_input_weight'],increment_weight=cfg['mpc_increment_weight'])
    R=par[21]+par[22]+tape[0,:,9];zero=np.zeros(n)
    obs=state_at(tape,0.,td,1);terms(x,obs,par)
    monitor_ma(x.copy(),previous,tape,td,0.,dt,fine,par)
    hard_hold(x,obs,par,dt)
    qp=SoftVOQP(n,par[6]);hardlogs=[];slacks=[];weightlogs=[];ttclogs=[]
    solve(x,goal,par[9],previous,warm,*mpc)
    hist=[];logs=[];oslog=[];idslog=[];costlog=[];times=[];marginlog=[]
    minima=np.full(3,1e20);path=0.;energy=0.;maxu=0.;status='timeout';started=time.perf_counter()
    for k in range(round(duration/dt)+1):
        t=k*dt;obs=state_at(tape,t,td,1)
        tick=time.perf_counter();nominal,plan,it,err=solve(x,goal,par[9],previous,warm,*mpc)
        warm[:-1]=plan[1:];warm[-1]=plan[-1]
        BV,CV,log,B,C,hardlog,weights,ttc=terms(x,obs,par)
        target=(nominal+par[6]*previous)/(1+par[6]);ids=np.full(n,-1,np.int64);calls=1
        margin=np.zeros(n)
        margin,certificate=hard_hold(x,obs,par,dt)
        certificate_available=bool(np.isfinite(margin).all())
        if certificate_available:u,ok,iters,delta=qp.solve(nominal,previous,BV,CV,B,C,margin,weights)
        else:u=np.full(3,np.nan);ok=False;iters=0;delta=np.full(n,np.nan)
        times.append(time.perf_counter()-tick)
        hardlog[:,12]=B@u+C if ok else np.full(n,np.nan)
        if save_history:
            hardlogs.append(hardlog);slacks.append(delta);weightlogs.append(weights);ttclogs.append(ttc)
        res=BV@u+CV if ok else np.full(n,np.nan);log[:,12]=res;log[:,8]=CV;log[:,9:12]=BV
        dist=np.linalg.norm(x[:3]-goal)
        hist.append(np.r_[t,x,u if ok else np.full(3,np.nan),nominal,dist,iters,np.linalg.norm(u-nominal),it,err])
        if save_history:
            logs.append(log);oslog.append(obs);idslog.append(ids);costlog.append(calls);marginlog.append(margin)
        body=log[:,0]-par[21]-obs[:,9]
        minima[:2]=np.minimum(minima[:2],[body.min(),hardlog[:,7].min()])
        if k==0 and (np.min(hardlog[:,6])<0 or hardlog[:,7].min()<0):
            status='initial_rejected';break
        if not certificate_available:status='sampled_certificate_unavailable';break
        if not ok:
            from scipy.optimize import linprog
            lp=linprog(np.zeros(3),A_ub=-B,b_ub=C-margin,bounds=[(-2,2)]*3,method='highs')
            status='qp_numerical_failure' if lp.success else 'qp_infeasible' if lp.status==2 else 'feasibility_numerical_uncertainty'
            break
        if body.min()<=0:status='collision';break
        if dist<.25 and np.linalg.norm(x[3:])<.2:status='goal';break
        if k==round(duration/dt):break
        maxu=max(maxu,np.max(np.abs(u)));energy+=dt*np.sum((u-nominal)**2)
        x,mins,length=monitor_ma(x,u,tape,td,t,dt,fine,par)
        minima=np.minimum(minima,mins);path+=length;previous=u.copy()
        if mins[0]<=0:status='collision';break
    hist=np.array(hist);dest.mkdir(parents=True,exist_ok=True)
    if save_history:
        np.savez_compressed(dest/'history.npz',history=hist,barriers=np.array(logs),obstacles=np.array(oslog),
            parameters=par,obstacle_tape=tape,tape_dt=td,motion_mode=1,workspace=scene['workspace'],
            initial=scene['start'],goal=goal,waypoints=scene['waypoints'],branches=np.array(idslog),
            controller_timing=np.array(times),branch_qp_calls=np.array(costlog),sampling_margins=np.array(marginlog),execution_residuals=np.array(hardlogs)[:,:,12]-np.array(marginlog),hard_barriers=np.array(hardlogs),vo_slack=np.array(slacks),vo_weights=np.array(weightlogs),ttc=np.array(ttclogs))
    row=dict(method_label=LABELS[name],scene='corridor_crossing',variant=name,termination=status,reached_goal=status=='goal',
        duration_s=float(hist[-1,0]),final_error_m=float(hist[-1,13]),collision=bool(minima[0]<=0),
        qp_failures=int(status in ('qp_infeasible','branch_set_infeasible')),numerical_failures=int(status=='qp_numerical_failure'),min_body_clearance_m=float(minima[0]),
        min_h=float(minima[1]),min_intersample_CBF_residual=float(minima[2]) if minima[2]<1e19 else None,
        max_acceleration_component=float(maxu),path_length_m=float(path),correction_integral=float(energy),
        obstacle_tape_sha256=hashlib.sha256(tape.tobytes()).hexdigest(),control_dt=dt,monitor_dt=fine,
        runtime_s=time.perf_counter()-started,config=cfg,park_kl=KL,park_km=KM,
        sampled_data_certificate_applied=True,
        sampled_numerical_padding=2e-7,
        execution_constraints='Hard braking B@u+C>=hold_margin; VO guidance has penalized nonnegative slack; common box +/-2',
        ma_settings=json.loads((HERE/'inputs/ma_settings.json').read_text(encoding='utf8')),
        qp_independent_solver_fallbacks=qp.fallbacks,
        sampling_scope='Hard braking barrier: outward interval bound through q=0; no hard constraint claimed for soft VO guidance.',
        timing_ms_p50=float(np.median(times)*1000),timing_ms_p95=float(np.quantile(times,.95)*1000),
        timing_ms_max=float(np.max(times)*1000),horizon_exhausted=status=='timeout',history_saved=bool(save_history))
    (dest/'metrics.json').write_text(json.dumps(row,indent=2),encoding='utf8')
    print(json.dumps({k:row[k] for k in ['variant','termination','duration_s','min_body_clearance_m','path_length_m','runtime_s']}),flush=True)
    return row
