"""Independent ULog + Gazebo-entity readback audit; never modifies flight data."""
from pathlib import Path
import sys,json,io,gzip,struct,hashlib,argparse
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'frozen_controller'))
from dynamic_study.kernel import constraints
from pyulog import ULog
def motion(tape,td,t):
    k=min(max(int((t+1e-10)/td),0),len(tape)-2);ds=max(0.,min(td,t-k*td));a=tape[k];b=tape[k+1];j=(b[:,6:9]-a[:,6:9])/td;o=a.copy()
    o[:,:3]=a[:,:3]+ds*a[:,3:6]+ds**2*a[:,6:9]/2+ds**3*j/6;o[:,3:6]=a[:,3:6]+ds*a[:,6:9]+ds**2*j/2;o[:,6:9]=a[:,6:9]+ds*j;return o
def check(folder):
    p=Path(folder);s=json.loads((p/'summary.json').read_text())
    with np.load(p/'history.npz',allow_pickle=False) as z:
        h=z['history'];boundary=z['boundary'];obs_log=z['obstacles'];actor=z['obstacle_pose_samples'];bar_raw=z['barriers'];physics=z['physics_truth'] if 'physics_truth' in z else np.empty((0,14))
    if not len(h):raise RuntimeError('No mission data; failure stays diagnostic')
    boundary=boundary[:len(h)];bar_raw=bar_raw[:len(h)]
    with (p/'obstacle_motion.bin').open('rb') as f:m,n,td=struct.unpack('<IId',f.read(16));tape=np.frombuffer(f.read(),dtype='<f8').reshape(m,n,10)
    ul=ULog(io.BytesIO(gzip.open(p/'flight.ulg.gz','rb').read()))
    def topic(name):return next(d.data for d in ul.data_list if d.name==name)
    tr=topic('vehicle_local_position_groundtruth');tt=tr['timestamp']*1e-6
    if len(physics):tt=physics[:,0]
    epoch=s['mission_sim_start'];times=h[:,0]+epoch
    mask=(tt>=times[0])&(tt<=float(s['mission_end_sim']));tt=tt[mask]
    pos=physics[mask,1:4] if len(physics) else np.c_[tr['y'],tr['x'],-tr['z']][mask]
    vel=physics[mask,4:7] if len(physics) else np.c_[tr['vy'],tr['vx'],-tr['vz']][mask]
    acc=physics[mask,7:10] if len(physics) else np.c_[tr['ay'],tr['ax'],-tr['az']][mask]
    woff=np.asarray(s['world_minus_px4_enu_offset']);xtrue=np.c_[pos-woff,vel]
    held=np.clip(np.searchsorted(times,tt,side='right')-1,0,len(times)-1)
    par=np.array(json.loads((ROOT/'frozen_controller/method_config.json').read_text())['parameters'])
    offset=obs_log[0,0,:3]-tape[0,0,:3]
    method=s.get('method','ra')
    sys.path.insert(0,str(ROOT/'frozen_baselines'))
    from baseline_coefficients import coefficients
    from ma_cbf_vo import terms
    phi=[];gaps=[];hs=[];ss=[];obs_used=[];radii=[];refs=[];diags=[];actor_error=[]
    expected_obs=np.array([motion(tape,td,t-epoch) for t in tt])
    grouped=[]
    for i in range(n):
        a=actor[actor[:,1]==i];order=np.argsort(a[:,0],kind='stable');a=a[order];_,index=np.unique(a[:,0],return_index=True);a=a[index]
        if not len(a):raise RuntimeError('No Gazebo entity readback for obstacle '+str(i))
        grouped.append(a)
        xyz=np.column_stack([np.interp(tt,a[:,0],a[:,j+2]) for j in range(3)])
        actor_error.extend(np.linalg.norm(xyz-(expected_obs[:,i,:3]+offset+woff),axis=1))
    for t,k,state,a in zip(tt,held,xtrue,acc):
        obs=motion(tape,td,t-epoch);obs[:,:3]+=offset
        for j,readback in enumerate(grouped):
            obs[j,:3]=[np.interp(t,readback[:,0],readback[:,2+l])-woff[l] for l in range(3)]
        ds=max(0.,t-times[k]);R,V,ref=boundary[k].T;omega=par[2];y=R-ref;b=V+omega*y;decay=np.exp(-omega*ds)
        rr=ref+(y+b*ds)*decay;vv=(V-omega*b*ds)*decay
        B,C,log=constraints(state,obs,rr.copy(),vv.copy(),ref.copy(),par,5) if method=='ra' else (terms(state,obs,par)[3:6] if method=='ma_cbf_vo' else coefficients(state,obs,rr.copy(),vv.copy(),ref.copy(),par,2 if method=='c3bf' else 8))
        phi.append(B@a+C);gaps.append(np.linalg.norm(obs[:,:3]-state[:3],axis=1)-par[21]-obs[:,9]);hs.append(log[:,7]);ss.append(log[:,6]);obs_used.append(obs);radii.append(rr);refs.append(ref);diags.append(log)
    phi=np.asarray(phi);gaps=np.asarray(gaps);hs=np.asarray(hs);ss=np.asarray(ss)
    geometry=json.loads((ROOT/'model_geometry.json').read_text())
    model_radius=geometry['validation_body_radius_m']
    model_gaps=gaps-(model_radius-par[21])
    elapsed=tt-epoch;distance=np.linalg.norm(xtrue[:,:3]-s['goal'],axis=1);arrivals=np.flatnonzero(distance<.25)
    first=float(elapsed[arrivals[0]]) if len(arrivals) else None
    pathend=arrivals[0]+1 if len(arrivals) else len(xtrue)
    path=float(np.linalg.norm(np.diff(xtrue[:pathend,:3],axis=0),axis=1).sum())
    command=h[held,10:13];error=np.linalg.norm(acc-command,axis=1)
    est=np.column_stack([np.interp(tt,times,h[:,4+j]) for j in range(6)])
    state_error=np.linalg.norm(est[:,:3]-xtrue[:,:3],axis=1)
    motors=topic('actuator_motors');mt=motors['timestamp']*1e-6;mi=(mt>=times[0])&(mt<=times[-1]);outputs=np.column_stack([motors['control[%d]'%j] for j in range(4)])
    saturation=int(np.sum(np.any(outputs[mi]>=.999,axis=1)))
    qa=topic('vehicle_attitude_groundtruth');qt=qa['timestamp']*1e-6;q=np.column_stack([np.interp(tt,qt,qa['q[%d]'%j]) for j in (1,2,3,0)])
    Rn=Rotation.from_quat(q).as_matrix();T=np.array([[0,1,0],[1,0,0],[0,0,-1]])
    quat=physics[mask,10:14] if len(physics) else Rotation.from_matrix(np.einsum('ij,njk,kl->nil',T,Rn,np.diag([1,-1,-1]))).as_quat()
    np.savez_compressed(p/'physical_audit_data.npz',time=elapsed,state=xtrue,acceleration=acc,command=command,phi=phi,gaps=model_gaps,controller_sphere_gaps=gaps,h=hs,s=ss,obstacles=np.asarray(obs_used),radius=np.asarray(radii),barriers=np.asarray(diags),quaternion=quat,controller_indices=held,estimated_state=est,parameters=par)
    result=dict(method=method,test=s['test'],completed=s['completed'],failure=s['failure'],first_arrival_estimate_s=s['first_arrival_s'],first_arrival_truth_s=first,
        path_to_truth_arrival_m=path,minimum_body_clearance_m=float(model_gaps.min()),minimum_controller_sphere_clearance_m=float(gaps.min()),validation_body_radius_m=model_radius,controller_body_radius_m=float(par[21]),minimum_actual_CBF_sampled=float(phi.min()),minimum_actual_h_sampled=float(hs.min()),minimum_actual_s_sampled=float(ss.min()),
        acceleration_tracking_peak_mps2=float(error.max()),acceleration_tracking_rms_mps2=float(np.sqrt(np.mean(error**2))),tracking_budget_mps2=.5,
        final_truth_goal_error_m=float(distance[-1]),final_truth_speed_mps=float(np.linalg.norm(vel[-1])),
        final_estimated_goal_error_m=float(np.linalg.norm(h[-1,4:7]-s['goal'])),final_estimated_speed_mps=float(np.linalg.norm(h[-1,7:10])),
        final_stopped=bool(distance[-1]<.25 and np.linalg.norm(vel[-1])<.2),
        maximum_position_estimation_error_m=float(state_error.max()),maximum_obstacle_entity_position_error_m=max(actor_error),
        actual_acceleration_source='Gazebo WorldLinearAcceleration physics component, ENU' if len(physics) else 'PX4 ULog Gazebo groundtruth ax/ay/az, generated internally from position differences',
        actual_CBF_scope='Independent physics-step saved truth samples, approximately 4ms; not a continuous-time proof' if len(physics) else 'Independent saved truth samples, approximately 20ms; not a continuous-time proof',
        physical_sample_count=len(tt),physical_median_sample_interval_s=float(np.median(np.diff(tt))),obstacle_entity_readback_count=len(actor),
        commanded_CBF_minimum=float(np.min(bar_raw[:,:,12])),motor_output_saturation_records=saturation,
        landed_and_disarmed=s['landed_and_disarmed'],repeated_state_steps=int(h[:,-2].sum()),
        control_p95_ms=float(np.quantile(h[:,19],.95)*1000),control_max_ms=float(h[:,19].max()*1000),overrun_steps=int(np.sum(h[:,19]>.02)),
        maximum_scheduling_lag_s=float(h[:,-1].max()),maximum_state_wall_age_s=float(h[:,3].max()),
        firmware_hash=ul.msg_info_dict.get('ver_sw'),model_autostart=ul.initial_parameters.get('SYS_AUTOSTART'))
    result['passed']=bool(s['completed'] and s['failure'] is None and first is not None and model_gaps.min()>0 and phi.min()>=-1e-6 and error.max()<=.5 and max(actor_error)<.02 and not saturation )
    result['terminal_policy']=s['terminal_policy'];result['landing_not_required']=True
    result['hashes']={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p/'history.npz',p/'flight.ulg.gz',p/'obstacle_motion.bin')}
    (p/'independent_check.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2),flush=True);return result
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('folder');a=ap.parse_args();check(a.folder)
