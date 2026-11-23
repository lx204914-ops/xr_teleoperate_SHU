"""Live SITL acceleration execution; all positions are produced by PX4/Gazebo."""
from pathlib import Path
import os,sys,time,argparse,json,hashlib,math,traceback,threading
os.environ['GZ_IP']='127.0.0.1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from pymavlink import mavutil
from first_arrival_platform import Platform,ROOT
from gz_bridge import Gazebo
from px4_controller import Px4Controller
from nominal_mpc import matrices,solve
from native_display_process import DisplayProcess as NativeOverlay
from runtime_optimizer import install as install_optimizer, stats as optimizer_stats
from performance_monitor import PerformanceMonitor
from periodic_publish75 import wait_until as wait_publication
from execution_response_profile import command as response_command
from continuous_instrumentation import TimedLink,TimedGazebo
from execution_runtime import install as install_execution_runtime
install_execution_runtime()
from equivalent_sparse_workspace import install as install_sparse_workspace
install_sparse_workspace()
install_optimizer()
from candidate_projection_backend import install as install_candidate_projection
install_candidate_projection()
DT=.02
NOMINAL_CRUISE=1.
def make_controller(method):
    if method=='ra':return Px4Controller(DT,nominal_cruise=NOMINAL_CRUISE)
    from baseline_adapter import BaselineController
    return BaselineController(method,nominal_cruise=NOMINAL_CRUISE)
def enu(v):return np.array([v[1],v[0],-v[2]],dtype=float)
def ned(v):return enu(v)
def tape_state(tape,td,t):
    k=min(max(int((t+1e-10)/td),0),len(tape)-2);ds=max(0.,min(td,t-k*td))
    a=tape[k];b=tape[k+1];j=(b[:,6:9]-a[:,6:9])/td;o=a.copy()
    o[:,:3]=a[:,:3]+ds*a[:,3:6]+ds**2*a[:,6:9]/2+ds**3*j/6
    o[:,3:6]=a[:,3:6]+ds*a[:,6:9]+ds**2*j/2;o[:,6:9]=a[:,6:9]+ds*j
    return o
from isolated_telemetry import IsolatedGazebo
Link=TimedLink;Gazebo=IsolatedGazebo

def main(args):
    global NOMINAL_CRUISE
    NOMINAL_CRUISE=args.cruise
    monitor=PerformanceMonitor()
    out=ROOT/'runs'/args.name
    os.environ['CBF_REAL_RUN_FOLDER']=str(out)
    failed_cycle=None
    ground_records=[]
    tracking_profile=None
    deadline_trace=[]
    scene_path=(args.scene if args.scene is not None else ROOT/'scenes/four_obstacle_scene.npz')
    inputs=list((ROOT/'code').glob('*.py'))+list((ROOT/'code').glob('*.cc'))+list((ROOT/'frozen_controller').rglob('*.py'))+list((ROOT/'frozen_controller').rglob('*.json'))
    inputs += [p for p in (ROOT/'execution_adaptation.json',ROOT/'execution_profile.json') if p.exists()]
    code_manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    tape=None;goal=None;scene_start=None;ctrl=None
    pilot=make_controller(args.method)
    mat=matrices(horizon=pilot.config['mpc_horizon'],dt=pilot.config['mpc_prediction_dt'],position_weight=pilot.config['mpc_position_weight'],velocity_weight=pilot.config['mpc_velocity_weight'],input_weight=pilot.config['mpc_input_weight'],increment_weight=pilot.config['mpc_increment_weight'])
    warm=np.zeros((pilot.config['mpc_horizon'],3));previous=np.zeros(3)
    solve(np.zeros(6),np.ones(3),2.,previous,warm,*mat)
    if args.test!='response':
        with np.load(scene_path,allow_pickle=False) as z:
            tape=z['tape'].copy();td=float(z['tape_dt']);scene_start=z['start'].copy();goal=z['goal'].copy()
        if args.test!='single' and args.duration>(len(tape)-1)*td+1e-9:
            raise ValueError('Requested flight exceeds frozen moving-obstacle tape; no extrapolated or frozen continuation')
        if args.test=='single':
            tape=tape[:,:1].copy();tape[:,:1,:3]=scene_start[:3]+np.array([5.,0.,0.]);tape[:,:1,3:9]=0.
        obs=tape_state(tape,td,0)
        temp=make_controller(args.method);temp.reset(scene_start,goal,obs);temp.initialize_execution(np.zeros(3));temp.execution_acceleration=np.zeros(3)
        _,status,_=temp.step(0.,scene_start,obs)
        if temp.last_result is not None and temp.last_result.control is not None:temp.advance_interval(scene_start,obs)
        from px4_adapter_base import transport_step
        transport_step(np.array([0.,1.,0.]),np.array([1.,0.,0.]),np.array([.99,.1,0.])/np.linalg.norm([.99,.1,0.]))
        import detour
        for axis in np.eye(3):detour.support(np.r_[np.eye(3),-np.eye(3)],-np.ones(6),axis)
        if args.method!='ra':
            warm_obs=obs.copy();warm_obs[:,:3]+=np.array([50.,0.,0.])
            from ma_cbf_vo import terms,hard_hold
            from sampled_baselines import hold_certificates
            from certificate_pool import candidate_lower
            terms(scene_start,warm_obs,temp.par);hard_hold(scene_start,warm_obs,temp.par,DT)
            hold_certificates(scene_start,warm_obs,temp.par,DT,2);candidate_lower(scene_start,warm_obs,temp.par,np.zeros(3),DT,.5,4)
        from sparse_projection import clarabel_projection
        clarabel_projection(np.zeros(3),np.r_[np.eye(3),-np.eye(3)],np.ones(6))
        from execution_runtime import warm_common_solver
        warm_common_solver(len(obs))
        print('WARMUP',status,flush=True)
    timing=[];rows=[];obstacle_rows=[];barrier_rows=[];boundary_rows=[];detours=[];nominals=[];computes=[]
    view=None
    mission_end_sim=None;terminal_truth=None;terminal_pause=False
    failure=None;arrival=None;landed=False;mission_start=None;link=None;gz=None;completed=False;world_offset=np.zeros(3);settled_since=None;two_second_check=None;settling_elapsed=None
    z=2.
    if tape is not None:z=max(z,float(scene_start[2]),float(1.+np.max(tape[:min(len(tape),2401),:,9]-tape[:min(len(tape),2401),:,2]+scene_start[2])))
    initial_offset=np.array([0.,0.,z])-scene_start[:3] if tape is not None else None
    with Platform(out,gui=args.gui,motion=tape,tape_dt=td if tape is not None else None,initial_offset=initial_offset):
      (out/'code_manifest.json').write_text(json.dumps(code_manifest,indent=2),encoding='utf8')
      try:
        gz=Gazebo();link=Link(args.execution)
        import inspect
        protocol=Path(inspect.getfile(type(link).__mro__[1]))
        (out/'communication_source_receipt.json').write_text(json.dumps(dict(active_base_class=type(link).__mro__[1].__module__,source=str(protocol),sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),requested_stream_intervals_us=link.requested_stream_intervals_us,actual_rate_measurement_required=True),indent=2))
        from flight_tracking_profile import apply,restore
        tracking_profile=apply(link,out)
        tic=time.monotonic()
        while (gz.sim is None or gz.odom is None or gz.physics is None) and time.monotonic()-tic<15:link.poll();time.sleep(.01)
        if gz.sim is None or gz.odom is None or gz.physics is None:raise RuntimeError('Gazebo clock/odometry/direct physics missing')
        from safe_preparation import execute as execute_preparation
        center=execute_preparation(link,z,gz,tape,initial_offset,out)
        (out/'native_gui_attach_ready').write_text('Physical hover established before GUI attachment')
        # Keep the verified physical hover during blocking GUI/JIT/GC setup.
        # This is pre-mission position control; it never generates navigation commands.
        prep_stop=threading.Event()
        def hold_preparation():
            while not prep_stop.is_set():
                link.poll();link.position_command(center);prep_stop.wait(.02)
        prep_thread=threading.Thread(target=hold_preparation,daemon=True);prep_thread.start()
        if link.acceleration is None:raise RuntimeError('No valid IMU/attitude acceleration feedback')
        world_offset=np.asarray(gz.physics[1:4])-link.position[:3]
        if tape is not None:
            offset=center-scene_start[:3];tape[:,:,:3]+=offset;goal=goal+offset
            obs=tape_state(tape,td,0)
            names=['ra_obstacle_'+str(i) for i in range(len(obs))]
            expected_motion=gz.motion_command(-1.,offset+world_offset)
            pause=time.monotonic()
            while time.monotonic()-pause<.25:
                link.poll();link.position_command(center);time.sleep(.02)
                if gz.motion_status is not None and np.max(np.abs(np.array(gz.motion_status)-expected_motion))<1e-8:break
            if gz.motion_status is None or np.max(np.abs(np.array(gz.motion_status)-expected_motion))>1e-8:raise RuntimeError('Obstacle motion epoch/offset not acknowledged')
            trim=np.zeros(3)
            ctrl=make_controller(args.method);ctrl.reset(link.position,goal,obs);ctrl.initialize_execution(trim);ctrl.execution_acceleration=link.acceleration.copy()
        else:goal=center.copy()
        if args.native_view:
            from dynamic_study.kernel import constraints as display_constraints
            _,_,display_barriers=(display_constraints(link.position,obs,ctrl.R.copy(),ctrl.V.copy(),ctrl.ref.copy(),ctrl.par,5) if args.method=='ra' else ctrl.raw_coefficients(link.position,obs))
            display_state=link.position.copy();display_state[:3]+=world_offset
            display_obstacles=obs.copy();display_obstacles[:,:3]+=world_offset
            prime=(display_state,np.asarray(gz.physics),display_obstacles,display_barriers,gz.sim)
            view=NativeOverlay(out,link.position[:3]+world_offset,goal+world_offset,ctrl.par,prime)
        from execution_runtime import request_scheduling
        request_scheduling(out)
        monitor.start_mission()
        # All blocking setup is complete BEFORE advancing the frozen scene.
        prep_stop.set();prep_thread.join(timeout=1.);link.poll()
        # Confirm that graphical initialization did not change the takeover state.
        center=link.takeoff(z)
        if args.initialize:
            init=json.loads((getattr(args,'initialization_file',None) or ROOT/'common_initialization.json').read_text());trim=np.asarray(init['command']);started=gz.sim;prep=[];lastsend=0.
            while gz.sim-started<init['duration_s']-1e-9:
                link.poll()
                if time.monotonic()-lastsend>=DT:
                    link.acceleration_command(trim);prep.append([gz.sim-started,*link.position,*link.acceleration,*gz.physics[7:10]]);lastsend=time.monotonic()
                time.sleep(.001)
            link.poll();(out/'physical_initialization.json').write_text(json.dumps(dict(policy=init,estimated_state=link.position.tolist(),actual_acceleration=gz.physics[7:10],preparation=prep),indent=2))
            if np.linalg.norm(link.acceleration-trim)>.5 or np.linalg.norm(np.asarray(gz.physics[7:10])-trim)>.5:raise RuntimeError('Common physical initialization tracking budget not met')
        if tape is not None:
            ctrl=make_controller(args.method);ctrl.reset(link.position,goal,obs);ctrl.initialize_execution(trim);ctrl.execution_acceleration=link.acceleration.copy()
        from affinity_isolation77 import isolate
        isolate(out)
        link.poll();mission_start=round(gz.sim/DT+1)*DT+.20
        if tape is not None:gz.motion_command(mission_start,offset+world_offset)
        wall_start=time.monotonic();k=0;last_posstamp=None;last_log=-1
        reference=np.zeros(3);reference_rate=np.zeros(3);tau=pilot.execution_time_scale
        while True:
            loop_wall=time.monotonic();link.poll();now=gz.sim;elapsed=now-mission_start
            if tape is not None and k>0 and elapsed>=0:
                actual=gz.physics.copy()
                # Termination observation only; simulator truth is never a control input.
                actual_obstacles=tape_state(tape,td,max(0.,float(actual[0])-mission_start))
                actual_gap=float(np.min(np.linalg.norm(actual_obstacles[:,:3]-(np.asarray(actual[1:4])-world_offset),axis=1)-actual_obstacles[:,9]-.4))
                if actual_gap<=0.:
                    mission_end_sim=float(actual[0]);terminal_truth=actual
                    failed_cycle=dict(kind='physical_collision_envelope',t=mission_end_sim-mission_start,minimum_body_clearance_m=actual_gap,validation_body_radius_m=.4,command_already_sent=False)
                    gz.set_paused(True);terminal_pause=True
                    raise RuntimeError('Physical obstacle collision envelope violated')
                error=float(np.linalg.norm(np.asarray(actual[1:4])-world_offset-goal))
                if error<.25:
                    arrival=float(actual[0]-mission_start);mission_end_sim=float(actual[0]);terminal_truth=actual;completed=True
                    gz.set_paused(True);terminal_pause=True
                    print('FIRST_TRUTH_ARRIVAL',arrival,'END_SITL_NO_LANDING',flush=True);break
            if link.mode!=6 or not link.armed:raise RuntimeError('Offboard/arming lost during acceleration task')
            if time.monotonic()-wall_start>(args.duration+25)*(5 if args.synchronous else 1):raise RuntimeError('Wall-clock watchdog')
            if elapsed+1e-9<k*DT:
                if k==0:
                    if args.initialize:link.acceleration_command(trim)
                    else:link.position_command(center)
                time.sleep(.001);continue
            age=time.monotonic()-link.wall_position
            if (now-link.position_stamp>.06) if args.synchronous else (age>.12):raise RuntimeError('Stale PX4 state')
            if not args.synchronous and time.monotonic()-link.acceleration_wall>.12:raise RuntimeError('Stale IMU acceleration feedback')
            if elapsed-k*DT>=DT:raise RuntimeError(f'Missed whole control period; no catch-up command burst: tick={k}, lag_ms={(elapsed-k*DT)*1000:.3f}')
            if gz.physics[3]<=.4:raise RuntimeError('Ground collision envelope violated')
            begin_wall=time.monotonic();position_wall=link.wall_position;position_stamp=link.position_stamp
            clock_at_begin=gz.clock_receipts[-1]
            tick_wall=clock_at_begin[0]-(clock_at_begin[1]-(mission_start+k*DT))
            t=k*DT;x=link.position.copy();monitor.begin(t);tic=time.perf_counter()
            if tape is None:
                if t<12:current_goal=center+np.array([.6*np.sin(2*np.pi*t/6),.5*np.sin(2*np.pi*t/8),.25*np.sin(2*np.pi*t/5)])
                else:current_goal=center
                nominal,plan,_,_=solve(x,current_goal,1.,previous,warm,*mat);warm[:-1]=plan[1:];warm[-1]=plan[-1]
                target=np.clip(nominal,-.8,.8);omega=1/tau;decay=np.exp(-omega*DT);y=reference-target;temp=reference_rate+omega*y
                reference=target+(y+temp*DT)*decay;reference_rate=(reference_rate-omega*temp*DT)*decay
                # The interface response test sends its actual smooth reference.
                # Do not feed measured acceleration back through an extra hard clamp.
                u=response_command(t,previous,DT,pilot.execution_time_scale);previous=u.copy();diag=None
            else:
                obs=tape_state(tape,td,elapsed)
                ctrl.execution_acceleration=link.acceleration.copy()
                ctrl.ground_minimum_estimated_z=.4-float(world_offset[2])
                u,status,diag=ctrl.step(t,x,obs)
                if getattr(ctrl,'ground_diagnostic',None) is not None:ground_records.append(dict(t=t,**ctrl.ground_diagnostic,execution_box=getattr(ctrl,'slew_diagnostic',None),command_vertical=None if u is None else float(u[2])))
                if u is None:
                    evaluation_end=time.monotonic();failed_compute=time.perf_counter()-tic
                    computes.append(failed_compute);monitor.end(failed_compute,diag,ctrl.last_result)
                    cc=clock_at_begin;failed_tick_wall=tick_wall
                    failed_cycle=dict(compute_ms=failed_compute*1000,begin_wall=begin_wall,tick_wall=failed_tick_wall,finish_phase_ms=(evaluation_end-failed_tick_wall)*1000,start_delay_ms=(begin_wall-failed_tick_wall)*1000,command_already_sent=False,kind='controller_no_input',t=t,state=x.tolist(),obstacles=obs.tolist(),R=ctrl.R.tolist(),V=ctrl.V.tolist(),Rref=ctrl.ref.tolist(),status=status,coefficients={key:diag[key].tolist() for key in ('B','C','sampling_margin','barriers') if key in diag})
                    raise RuntimeError('Safety controller: '+status)
                obstacle_rows.append(obs.copy());barrier_rows.append(diag['barriers'].copy());boundary_rows.append(np.c_[ctrl.R,ctrl.V,ctrl.ref])
                detours.append(diag['detour']);nominal=diag['nominal']
            compute=time.perf_counter()-tic;computes.append(compute);monitor.end(compute,diag,ctrl.last_result if ctrl is not None else None)
            # Keep a constant publish phase between the 4ms PX4/Gazebo updates.
            # The physical world remains unpaused throughout this bounded wait.
            clock_now=clock_at_begin
            # Deadline anchor is immutable during evaluation; late clock callbacks cannot move it.
            publish_wall=tick_wall+.014
            publication_wait=wait_publication(publish_wall)
            decision_wall=time.monotonic()
            hard_deadline=min(tick_wall+DT,loop_wall+DT)
            if decision_wall>hard_deadline-.0005:
                failed_cycle=dict(t=t,loop_wall=loop_wall,begin_wall=begin_wall,tick_wall=tick_wall,compute_ms=compute*1000,decision_wall=decision_wall,publish_wall=publish_wall,start_delay_ms=(begin_wall-tick_wall)*1000,finish_phase_ms=(decision_wall-tick_wall)*1000,phase_error_ms=(decision_wall-publish_wall)*1000,clock_receipt=clock_now,state=x.tolist(),obstacles=[] if tape is None else obs.tolist(),command_not_sent=u.tolist(),R=[] if ctrl is None else ctrl.R.tolist(),V=[] if ctrl is None else ctrl.V.tolist(),Rref=[] if ctrl is None else ctrl.ref.tolist())
                raise RuntimeError('Complete 20ms cycle deadline has insufficient send reserve; no catch-up command')
            before_send=time.monotonic();link.acceleration_command(u);after_send=time.monotonic()
            send_sim=gz.sim;send_record=link.send_trace[-1];send_record.update(controller_index=k,controller_t=t,state_stamp=position_stamp,world_sim_snapshot=send_sim)
            truth=gz.physics.copy();repeat=float(last_posstamp==link.position_stamp);last_posstamp=link.position_stamp
            rows.append([elapsed,t,link.position_stamp,age,*x,*u,*link.acceleration,*nominal,compute,repeat,now-mission_start-k*DT])
            nominals.append(nominal)
            if tape is not None:
                ctrl.advance_interval(x,obs)
                gap=float(np.min(np.linalg.norm(obs[:,:3]-(np.array(truth[1:4])-world_offset),axis=1)-.4-obs[:,9]))
                if view is not None:
                    display_obs=obs.copy();display_obs[:,:3]+=world_offset
                    display_state=x.copy();display_state[:3]+=world_offset
                    view.update(t,display_state,truth,display_obs,diag['barriers'],u,gap,mission_start+t)
                cycle_finish=time.monotonic()
                timing.append([k,t,elapsed,loop_wall,begin_wall,position_wall,position_stamp,before_send,after_send,cycle_finish,send_sim])
                deadline_trace.append(dict(t=t,preferred_phase_ms=14.,hard_budget_ms=20.,anchor='clock_snapshot_before_evaluation',anchor_receipt=clock_at_begin,start_delay_ms=(begin_wall-tick_wall)*1000,send_phase_ms=(after_send-tick_wall)*1000,finish_phase_ms=(cycle_finish-tick_wall)*1000,publication_wait_cpu_ms=publication_wait["cpu_ms"],publication_wait_wall_ms=publication_wait["wall_ms"],passed=cycle_finish<=hard_deadline))
                if cycle_finish>hard_deadline:
                    failed_cycle=dict(kind='post_send_full_cycle_deadline',t=t,compute_ms=compute*1000,start_delay_ms=(begin_wall-tick_wall)*1000,finish_phase_ms=(cycle_finish-tick_wall)*1000,phase_error_ms=(cycle_finish-hard_deadline)*1000,command_already_sent=True)
                    raise RuntimeError('Complete active control cycle exceeded 20ms')
                if gap<=0:raise RuntimeError('Collision envelope violated')
            else:
                cycle_finish=time.monotonic()
                timing.append([k,t,elapsed,loop_wall,begin_wall,position_wall,position_stamp,before_send,after_send,cycle_finish,send_sim])
                deadline_trace.append(dict(t=t,preferred_phase_ms=14.,hard_budget_ms=20.,anchor='clock_snapshot_before_evaluation',anchor_receipt=clock_at_begin,start_delay_ms=(begin_wall-tick_wall)*1000,send_phase_ms=(after_send-tick_wall)*1000,finish_phase_ms=(cycle_finish-tick_wall)*1000,publication_wait_cpu_ms=publication_wait["cpu_ms"],publication_wait_wall_ms=publication_wait["wall_ms"],passed=cycle_finish<=hard_deadline))
                if cycle_finish>hard_deadline:
                    failed_cycle=dict(kind='response_post_send_full_cycle_deadline',t=t,compute_ms=compute*1000,finish_phase_ms=(cycle_finish-tick_wall)*1000,command_already_sent=True)
                    raise RuntimeError('Complete response cycle exceeded 20ms')
                if t>=args.duration:completed=True;break
            if int(t)//2>last_log:
                last_log=int(t)//2;print(json.dumps(dict(t=t,p=x[:3].tolist(),speed=float(np.linalg.norm(x[3:])),goal_error=float(np.linalg.norm(x[:3]-goal)),arrival=arrival,compute_ms=compute*1000)),flush=True)
            if (ROOT/'STOP_AFTER_LANDING').exists():break
            if t>=args.duration:break
            k+=1
            if args.synchronous:
                pace=wall_start+k*DT/args.wall_rtf-time.monotonic()
                if pace>0:time.sleep(pace)
                previous_sim=gz.sim;time.sleep(.003);gz.step(round(DT/.004));until=time.monotonic()+3
                while (gz.sim<previous_sim+DT-1e-8) and time.monotonic()<until:link.poll();time.sleep(.001)
                if gz.sim<previous_sim+DT-1e-8:raise RuntimeError('World did not complete requested physics steps')
                if abs(gz.sim-previous_sim-DT)>1e-7:raise RuntimeError('World advanced a different interval than requested')
                time.sleep(.002)
      except Exception as exc:
        failure=str(exc)
        if failed_cycle is None and mission_start is not None and 'loop_wall' in locals():
            stop_wall=time.monotonic();cc=gz.clock_receipts[-1];planned=cc[0]-(cc[1]-(mission_start+k*DT))
            failed_cycle=dict(kind='runner_exception',t=k*DT,finish_phase_ms=(stop_wall-planned)*1000,loop_wall=loop_wall,stop_wall=stop_wall,error=failure)
        print('FLIGHT_ERROR',failure,flush=True);traceback.print_exc()
      finally:
        if mission_start is None and gz is not None:
            try:gz.set_paused(True);terminal_pause=True
            except Exception:pass
        if mission_start is not None and gz is not None:
            if mission_end_sim is None:mission_end_sim=float(gz.physics[0]);terminal_truth=gz.physics.copy()
            if not completed and failure is None:failure='observation_horizon_expired'
            if not terminal_pause:
                try:gz.set_paused(True);terminal_pause=True
                except Exception:pass
        if terminal_pause and link is not None:
            drain_begin=time.monotonic();received_before=len(link.receive_trace)
            while time.monotonic()-drain_begin<.12:
                link.poll();time.sleep(.002)
            (out/'post_pause_receive_drain.json').write_text(json.dumps(dict(physical_world_paused=True,elapsed_wall_s=time.monotonic()-drain_begin,additional_messages=len(link.receive_trace)-received_before,new_control_sent=False,physics_advance_requested=False),indent=2))
        if view is not None:view.close(completed,failure)
        if 'prep_stop' in locals():prep_stop.set()
        # No landing command. Parameter files are restored after virtual SITL exits.
        if gz is not None and hasattr(gz,'finish_actor_logging'):gz.finish_actor_logging()
        (out/'ground_execution_rows.json').write_text(json.dumps(ground_records))
        if failed_cycle is not None:(out/'failed_cycle.json').write_text(json.dumps(failed_cycle,indent=2))
        (out/'cycle_deadline_trace.json').write_text(json.dumps(dict(control_period_ms=20.,preferred_publication_phase_ms=14.,hard_cycle_budget_ms=20.,pre_send_reserve_ms=.5,rows=deadline_trace),indent=2))
        (out/'execution_trace.json').write_text(json.dumps(dict(timing_columns=['index','nominal_sim_t','observed_elapsed','loop_wall','begin_wall','state_receive_wall','state_stamp','before_send_wall','after_send_wall','cycle_finish_wall','send_sim_snapshot'],timing=timing,send=link.send_trace if link else [],receive=link.receive_trace if link else [],timesync=link.timesync if link else [],clock_receipts=gz.clock_receipts if gz else [],physics_receipts=gz.physics_receipts if gz else [])),encoding='utf8')
        np.savez_compressed(out/'history.npz',history=np.asarray(rows),obstacles=np.asarray(obstacle_rows),barriers=np.asarray(barrier_rows),boundary=np.asarray(boundary_rows),goal=goal if goal is not None else [],truth_odometry=np.asarray(gz.samples) if gz else [],physics_truth=np.asarray(gz.physics_samples) if gz else [],obstacle_pose_samples=np.asarray(gz.obstacle_pose_samples) if gz else [],px4_position=np.asarray(link.position_records) if link else [],imu_acceleration=np.asarray(link.acceleration_records) if link else [],
            columns=np.array(['sim_elapsed','controller_t','px4_stamp','state_wall_age','px','py','pz','vx','vy','vz','ux','uy','uz','estimated_ax','estimated_ay','estimated_az','nominal_x','nominal_y','nominal_z','compute_s','repeated_state','scheduling_lag']))
        (out/'events.json').write_text(json.dumps(detours,default=lambda v:v.tolist() if hasattr(v,'tolist') else float(v)),encoding='utf8')
        summary=dict(terminal_policy='first_truth_entry_stop_SITL_no_landing',arrival_radius_m=.25,mission_end_sim=mission_end_sim,terminal_truth=terminal_truth,post_terminal_world_paused=terminal_pause,transport_partition=os.environ.get('GZ_PARTITION'),observation_horizon_s=args.duration,method=args.method,test=args.test,completed=completed,failure=failure,first_arrival_s=arrival,post_arrival_2s_check=None,settling_elapsed_after_first_entry_s=0.,settled_dwell_requirement_s=0.,landed_and_disarmed=landed,control_dt_s=DT,rows=len(rows),SITL_only=True,controller_state='PX4 LOCAL_POSITION_NED estimate',actual_trajectory='Gazebo direct physics position/velocity/acceleration components',acceleration_feedback='PX4 HIGHRES_IMU de-biased specific force rotated by ATTITUDE_QUATERNION; gravity compensated; causal 5ms low pass',obstacles='prescribed Gazebo kinematic sphere models',mission_sim_start=mission_start,
            goal=goal.tolist() if goal is not None else None,control_p95_ms=1000*float(np.quantile(computes,.95)) if computes else None,
            world_minus_px4_enu_offset=world_offset.tolist(),execution_reference_time_scale_s=pilot.execution_time_scale,nominal_cruise_mps=float(pilot.config['parameters'][9]),nominal_mpc_component_limit_mps2=pilot.mpc_input_limit,physical_execution_component_limit_mps2=1.5,execution_policy=pilot.execution_policy,scene_file=str(scene_path) if tape is not None else None,scene_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest() if tape is not None else None,execution_interface=args.execution,downstream_payload_is_internal_tracking_reference=True,synchronous_physics=args.synchronous,world_step_ack_lost=gz.step_ack_lost if gz else None,hard_realtime_accepted=False,continuous_world=True,no_catchup_bursts=True,command_mask=3079 if args.execution=='velocity_ff' else 3135,force_flag=False,acknowledgments=link.acks if link else [],messages=link.text if link else [])
        monitor.save(out)
        if gz is not None:gz.rpc.close()
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ('messages','acknowledgments')}),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--name',required=True);ap.add_argument('--cruise',type=float,default=1.);ap.add_argument('--native-view',action='store_true');ap.add_argument('--wall-rtf',type=float,default=1.);ap.add_argument('--initialize',action='store_true');ap.add_argument('--method',choices=['ra','ma_cbf_vo','c3bf','park'],default='ra');ap.add_argument('--test',choices=['response','single','four','ten','fifty','hundred'],default='response');ap.add_argument('--scene',type=Path);ap.add_argument('--duration',type=float,default=18.);ap.add_argument('--execution',choices=['acceleration','velocity_ff'],default='acceleration');ap.add_argument('--synchronous',action='store_true');ap.add_argument('--gui',action='store_true');parsed=ap.parse_args();assert not parsed.synchronous,'Continuous runner forbids world stepping';main(parsed)
