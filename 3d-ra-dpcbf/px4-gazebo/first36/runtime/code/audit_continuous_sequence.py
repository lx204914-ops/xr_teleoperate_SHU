"""Read-only continuous execution audit using PX4 received setpoints and physics."""
from pathlib import Path
import json,hashlib,sys,gzip,io,struct,csv
import numpy as np
from pyulog import ULog
ROOT=Path(__file__).resolve().parents[1]

def stats(x):
    a=np.asarray(x,dtype=float);a=a[np.isfinite(a)]
    return dict(count=len(a),min=float(a.min()),p50=float(np.quantile(a,.5)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else None

def motion(tape,td,t):
    k=min(max(int((t+1e-10)/td),0),len(tape)-2);dt=np.clip(t-k*td,0,td);a=tape[k];j=(tape[k+1,:,6:9]-a[:,6:9])/td;b=a.copy()
    b[:,:3]+=dt*a[:,3:6]+.5*dt**2*a[:,6:9]+dt**3*j/6;b[:,3:6]+=dt*a[:,6:9]+.5*dt**2*j;b[:,6:9]+=dt*j;return b

def check(folder):
    p=Path(folder);s=json.loads((p/'summary.json').read_text());trace=json.loads((p/'execution_trace.json').read_text());freeze=json.loads((ROOT/'frozen_baseline.json').read_text())
    unchanged=[]
    for key,value in freeze['main_method_source_lock'].items():
        assert hashlib.sha256((ROOT/key.replace(chr(92),'/')).read_bytes()).hexdigest()==value,key;unchanged.append(key)
    assert s['scene_sha256'] in [v['sha256'] for v in freeze['scene_cases'].values()] and not s['synchronous_physics']
    with np.load(p/'history.npz',allow_pickle=False) as z:data={k:z[k] for k in z.files}
    h=data['history'];boundary=data['boundary'];physics=data['physics_truth'];actors=data['obstacle_pose_samples']
    unexecuted_evaluation_rows=len(boundary)-len(h);boundary=boundary[:len(h)]
    if h.ndim!=2 or not len(h):raise RuntimeError('No physical mission samples to audit')
    ul=ULog(io.BytesIO(gzip.open(p/'flight.ulg.gz','rb').read()))
    topic=lambda name:next(d.data for d in ul.data_list if d.name==name)
    rx=topic('trajectory_setpoint');rx_t=rx['timestamp']*1e-6
    vec=lambda prefix:np.c_[rx[prefix+'[1]'],rx[prefix+'[0]'],-rx[prefix+'[2]']]
    rx_v=vec('velocity');rx_a=vec('acceleration')
    # PX4 correctly sets ignored velocity fields to NaN in acceleration-only mode.
    # Match only the active acceleration payload; do not reject valid commands.
    if s['execution_interface']=='acceleration':rx_v=np.zeros_like(rx_v)
    payload=np.c_[rx_v,rx_a]
    sent=[row for row in trace['send'] if 'controller_index' in row]
    sends=np.array([r['wall_end'] for r in sent]);vectors=np.array([r['velocity_enu']+r['acceleration_enu'] for r in sent])
    clock=np.array(trace['clock_receipts']);order=np.argsort(clock[:,0]);clock=clock[order]
    sim_send=np.interp(sends,clock[:,0],clock[:,1]);matched=[]
    # Match the values actually serialized on the MAVLink wire. Nearest-value
    # matching can reuse a single command when two inputs round to the same
    # float32 payload. Preserve chronological order and require unique receipts.
    wire=np.asarray(vectors,dtype=np.float32);lookup={}
    for j,row in enumerate(np.asarray(payload,dtype=np.float32)):
        if not np.isfinite(row).all():continue
        if not s['mission_sim_start']-.02<rx_t[j]<s['mission_sim_start']+s['rows']*.02+.1:continue
        lookup.setdefault(tuple(row.tolist()),[]).append(j)
    previous=-1;first={}
    for i,row in enumerate(wire):
        candidates=lookup.get(tuple(row.tolist()),[])
        valid=[j for j in candidates if j>previous and sent[i]['state_stamp']-.004<=rx_t[j]<=sent[i]['state_stamp']+.08]
        if not valid:continue
        j=valid[0];previous=j;error=float(np.max(abs(vectors[i]-payload[j])))
        matched.append((i,j,error));first[i]=(j,error)
    accepted_ids=sorted(first);accepted_times=np.array([rx_t[first[i][0]] for i in accepted_ids]);accepted_commands=np.array([sent[i]['acceleration_enu'] for i in accepted_ids])
    assert not len(accepted_times)>1 or np.all(np.diff(accepted_times)>0)
    if not len(accepted_ids):raise RuntimeError('No received command could be matched to a real sent command')
    mismatch=[i for i in range(len(sent)) if i not in first]
    timing=np.asarray(trace['timing']);epoch=s['mission_sim_start'];offset=np.asarray(s['world_minus_px4_enu_offset'])
    mission_end=epoch+float(h[-1,0]);select=(physics[:,0]>=accepted_times[0])&(physics[:,0]<=mission_end);physical=physics[select]
    tt=physical[:,0];state=np.c_[physical[:,1:4]-offset,physical[:,4:7]];acc=physical[:,7:10]
    cmd_index=np.clip(np.searchsorted(accepted_times,tt,side='right')-1,0,len(accepted_times)-1);command=accepted_commands[cmd_index]
    error=np.linalg.norm(acc-command,axis=1)
    # PX4 hrt advances on sensor steps; Gazebo publishes its clock separately.
    # Cover two 4ms phase bins instead of claiming perfect cross-clock alignment.
    conservative_error=error.copy()
    for phase in (.004,.008):
        ix=np.clip(np.searchsorted(accepted_times+phase,tt,side='right')-1,0,len(accepted_times)-1)
        conservative_error=np.maximum(conservative_error,np.linalg.norm(acc-accepted_commands[ix],axis=1))
    schedule_index=np.clip(np.searchsorted(s['mission_sim_start']+h[:,0],tt,side='right')-1,0,len(h)-1)
    conservative_error=np.maximum(conservative_error,np.linalg.norm(acc-h[schedule_index,10:13],axis=1))
    # Radius snapshot belongs to control evaluation; use the measured observation time,
    # and independently report any nominal-vs-physical integration mismatch.
    eval_times=epoch+h[:,0];held=np.clip(np.searchsorted(eval_times,tt,side='right')-1,0,len(h)-1)
    R=boundary[held,:,0];V=boundary[held,:,1];ref=boundary[held,:,2]
    par=np.asarray(json.loads((ROOT/'frozen_controller/method_config.json').read_text())['parameters']);omega=par[2]
    elapsed=(tt-eval_times[held])[:,None];y=R-ref;b=V+omega*y;decay=np.exp(-omega*elapsed)
    radius=ref+(y+b*elapsed)*decay;rate=(V-omega*b*elapsed)*decay
    with (p/'obstacle_motion.bin').open('rb') as f:
        m,n,td=struct.unpack('<IId',f.read(16));tape=np.frombuffer(f.read(),dtype='<f8').reshape(m,n,10)
    scene=np.load(s['scene_file'],allow_pickle=False);shift=np.asarray(s['goal'])-scene['goal'];scene.close()
    obs=np.array([motion(tape,td,t-epoch) for t in tt]);obs[:,:,:3]+=shift
    entity_error=[]
    for i in range(n):
        a=actors[actors[:,1]==i];a=a[np.argsort(a[:,0],kind='stable')];_,indices=np.unique(a[:,0],return_index=True);a=a[indices]
        assert len(a)>0,'Missing obstacle entity readback'
        xyz=np.column_stack([np.interp(tt,a[:,0],a[:,j+2]) for j in range(3)])-offset
        entity_error.extend(np.linalg.norm(xyz-obs[:,i,:3],axis=1));obs[:,i,:3]=xyz
    r=obs[:,:,:3]-state[:,None,:3];D=np.linalg.norm(r,axis=2);e=r/D[:,:,None];w=obs[:,:,3:6]-state[:,None,3:];q=np.sum(e*w,axis=2);z=w-q[:,:,None]*e;Z=np.sum(z*z,axis=2)
    beta=par[1]/(par[0]*par[19]);clear=D-radius;K=1/par[0]+beta*Z;barrier=q-rate+K*clear;rdd=omega**2*(ref-radius)-2*omega*rate
    B=-e-2*beta*clear[:,:,None]*z
    A=Z/D+np.sum(e*obs[:,:,6:9],axis=2)-rdd+K*(q-rate)+2*beta*clear*np.sum(z*obs[:,:,6:9],axis=2)-2*beta*clear*q*Z/D
    phi=A+par[3]*barrier+np.sum(B*acc[:,None,:],axis=2)-par[16]*np.linalg.norm(B,axis=2)
    body=json.loads((ROOT/'model_geometry.json').read_text())['validation_body_radius_m'];gaps=D-body-obs[:,:,9]
    distances=np.linalg.norm(state[:,:3]-s['goal'],axis=1);arrival=np.flatnonzero(distances<.25);arrival_s=float(tt[arrival[0]]-epoch) if len(arrival) else None
    work=(timing[:,9]-timing[:,3])*1000;solve=(timing[:,7]-timing[:,4])*1000
    clock_mission=clock[(clock[:,1]>=epoch)&(clock[:,1]<=mission_end)]
    rtf=(clock_mission[-1,1]-clock_mission[0,1])/(clock_mission[-1,0]-clock_mission[0,0])
    communication=(accepted_times-sim_send[accepted_ids])*1000
    # Sim clock is published every 4ms. Negative sub-step interpolation estimates
    # remain in the report; never clip them to fabricate a positive latency.
    e2e=np.array([(accepted_times[k]-sent[i]['state_stamp'])*1000 for k,i in enumerate(accepted_ids)])
    # Exclude packets drained after mission termination/GUI close, not their values.
    # The complete trace still retains those packets and their actual queue delay.
    telemetry=[v for v in trace['receive'] if v['kind']=='LOCAL_POSITION_NED' and epoch<=v['stamp']<=mission_end and timing[0,3]<=v['wall']<=sends[-1]]
    tele_age=np.array([(np.interp(v['wall'],clock[:,0],clock[:,1])-v['stamp'])*1000 for v in telemetry])
    rtt=[v['rtt_ms'] for v in trace['timesync'] if clock_mission[0,0]<=v['wall']<=clock_mission[-1,0]]
    motor=topic('actuator_motors');mi=(motor['timestamp']*1e-6>=epoch)&(motor['timestamp']*1e-6<=mission_end)
    saturation=int(np.sum(np.any(np.column_stack([motor['control[%d]'%j][mi] for j in range(4)])>=.999,axis=1)))
    dt=np.diff(h[:,0]);R0=boundary[:-1,:,0];V0=boundary[:-1,:,1];ref0=boundary[:-1,:,2];bb=V0+omega*(R0-ref0);dd=np.exp(-omega*dt[:,None]);pred=ref0+(R0-ref0+bb*dt[:,None])*dd
    result=dict(unexecuted_evaluation_rows=unexecuted_evaluation_rows,run=p.name,completed=s['completed'],failure=s['failure'],main_source_files_unchanged=len(unchanged),scene_sha256=s['scene_sha256'],
        first_arrival_truth_s=arrival_s,landed_and_disarmed=s['landed_and_disarmed'],minimum_body_clearance_m=float(gaps.min()),actual_sampled_CBF_min=float(phi.min()),
        tracking_peak_mps2=float(error.max()),tracking_peak_conservative_mps2=float(conservative_error.max()),tracking_rms_mps2=float(np.sqrt(np.mean(error**2))),tracking_budget_mps2=.5,
        tracking_alignment_note='Conservative peak covers the scheduled command and received-command phase shifts of 0, 4 and 8ms; raw acceptance alignment reported separately.',
        final_goal_error_m=float(distances[-1]),final_speed_mps=float(np.linalg.norm(state[-1,3:])),
        control_compute_ms=stats(h[:,19]*1000),complete_active_cycle_ms=stats(work),state_snapshot_to_send_ms=stats((timing[:,8]-timing[:,4])*1000),
        command_send_interval_wall_ms=stats(np.diff(sends)*1000),command_accept_interval_sim_ms=stats(np.diff(accepted_times)*1000),
        state_publication_to_PX4_accept_ms=stats(e2e),state_publication_to_app_receive_ms_estimate=stats(tele_age),
        isolated_one_way_network_delay_resolved=False,timesync_roundtrip_wall_ms=stats(rtt),
        clock_domain_diagnostic_unaligned_send_to_receive_ms=stats(communication),
        latency_resolution_note='End-to-end state publication to command acceptance uses the SAME PX4 clock. TIMESYNC round-trip uses the SAME host monotonic clock. Gazebo and PX4 clock phases differ by a physics step, so cross-clock negative differences are diagnostic only and NOT network latency. No isolated one-way delay is claimed.',
        sent_commands=len(sent),matched_received_commands=len(accepted_ids),unmatched_command_indices=mismatch,max_payload_error=float(max(e for _,_,e in matched)),
        actual_world_real_time_factor=float(rtf),physics_step_s=stats(np.diff(tt)),maximum_control_schedule_lag_ms=float(h[:,-1].max()*1000),
        full_active_cycles_over_20ms=int(np.sum(work>20)),solver_steps_over_20ms=int(np.sum(h[:,19]>.02)),
        nominal_vs_physical_radius_step_difference_m=float(np.max(abs(pred-boundary[1:,:,0]))),
        obstacle_readback_error_m=float(max(entity_error)),motor_saturation_records=saturation,world_pause_calls_during_mission=0,world_multistep_calls=0,
        timestamp_domain='PX4 SITL hrt and Gazebo sim clock; host monotonic clock recorded separately',
        hard_realtime_guarantee=False,scope='One finite free-running SITL case, truth obstacles; no perception or real-aircraft claim')
    phase_trace=json.loads((p/'cycle_deadline_trace.json').read_text());phase_values=[v['finish_phase_ms'] for v in phase_trace['rows']]
    failure_path=p/'failed_cycle.json'
    failed=json.loads(failure_path.read_text()) if failure_path.exists() else None
    if failed is not None and failed.get('finish_phase_ms') is not None:phase_values.append(failed['finish_phase_ms'])
    result['matching_policy']='float32_wire_payload_sequence; unique chronological receipts within original state-publication clock domain'
    result['complete_planned_cycle_ms']=stats(phase_values)
    result['failed_evaluation_timing_recorded']=None if failed is None else failed.get('finish_phase_ms') is not None
    result['planned_cycles_over_20ms']=sum(v>20 for v in phase_values)
    result['passed']=bool(not result['planned_cycles_over_20ms'] and s['completed'] and s['failure'] is None and arrival_s is not None and s['landed_and_disarmed'] and
        conservative_error.max()<=.5 and gaps.min()>0 and phi.min()>=-1e-6 and not saturation and not mismatch and np.max(work)<=20 and abs(rtf-1)<.05 and distances[-1]<.25 and np.linalg.norm(state[-1,3:])<.2)
    np.savez_compressed(p/'continuous_sequence_audit_data.npz',time=tt-epoch,state=state,acceleration=acc,received_command=command,tracking_error=error,phi=phi,gaps=gaps,radius=radius,
        command_send_sim_estimate=sim_send,command_receive_sim=accepted_times,command_received_indices=accepted_ids)
    (p/'continuous_sequence_check.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    with (p/'control_timing_sequence.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.writer(f);writer.writerow(trace['timing_columns']);writer.writerows(trace['timing'])
    print(json.dumps(result,indent=2));return result

if __name__=='__main__':check(Path(sys.argv[1]))
