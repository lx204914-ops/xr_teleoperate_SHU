"""Own and stop only one PX4/Gazebo SITL process group; no aircraft interfaces."""
from pathlib import Path
import os,subprocess,time,signal,gzip,shutil,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
PX4=Path(os.environ.get('CBF_PX4_ROOT',str(Path.home()/'cbf_px4/PX4-Autopilot')))
PY=os.environ.get('CBF_PYTHON',str(Path.home()/'cbf_px4/.venv/bin/python'))
FIRMWARE=ROOT/'build/feedback_receipt_bin/px4'
class Platform:
    def __init__(self,out,gui=False,motion=None,tape_dt=None,initial_offset=None):
        self.out=Path(out);self.gui=gui;self.proc=None;self.motion=motion;self.tape_dt=tape_dt;self.initial_offset=initial_offset
    def __enter__(self):
        if subprocess.run(['pgrep','-x','px4'],capture_output=True).returncode==0:raise RuntimeError('Existing PX4 preserved')
        self.out.mkdir(parents=True,exist_ok=False)
        if (ROOT/'stage_lock.json').exists():(self.out/'stage_lock_at_run.json').write_bytes((ROOT/'stage_lock.json').read_bytes())
        if (ROOT/'execution_contract.json').exists():(self.out/'execution_contract.json').write_bytes((ROOT/'execution_contract.json').read_bytes())
        world=ROOT/'worlds/ra_lab.sdf';world.parent.mkdir(exist_ok=True)
        src=(PX4/'Tools/simulation/gz/worlds/default.sdf').read_text(encoding='utf-8-sig').replace('<world name="default">','<world name="ra_lab">')
        # Additional publisher only; official X500 dynamics/actuators/sensors unchanged.
        model='''<include><uri>model://x500</uri><name>x500_0</name><pose>0 0 0.25 0 0 0</pose>
        <plugin filename="gz-sim-odometry-publisher-system" name="gz::sim::systems::OdometryPublisher">
        <odom_publish_frequency>250</odom_publish_frequency><dimensions>3</dimensions>
        <odom_topic>/ra/odometry</odom_topic><odom_frame>world</odom_frame><robot_base_frame>base_link</robot_base_frame>
        </plugin></include>'''
        # Explicit observer/motion plugins require retaining the active defaults.
        import xml.etree.ElementTree as ET,copy
        defaults=ET.parse(PX4/'src/modules/simulation/gz_bridge/server.config').getroot()
        for plugin in defaults.findall('./plugins/plugin'):
            if plugin.get('entity_type')=='world':
                element=copy.deepcopy(plugin);element.attrib.pop('entity_name',None);element.attrib.pop('entity_type',None)
                model+=ET.tostring(element,encoding='unicode')
        library=ROOT/'build/libra_obstacle_motion.so'
        model+=f'<plugin filename="{library}" name="RaVehicleObserver"/>'
        if self.motion is not None:
            import numpy as np,struct
            datafile=self.out/'obstacle_motion.bin'
            with datafile.open('wb') as f:
                f.write(struct.pack('<IId',len(self.motion),self.motion.shape[1],self.tape_dt));f.write(np.asarray(self.motion,dtype='<f8').tobytes())
            off=self.initial_offset
            for i,o in enumerate(self.motion[0]):
                x,y,z=o[:3]+off;r=float(o[9]);name='ra_obstacle_'+str(i)
                model+=f'''<model name="{name}"><static>true</static><pose>{x} {y} {z} 0 0 0</pose><link name="body"><collision name="collision"><geometry><sphere><radius>{r}</radius></sphere></geometry></collision><visual name="visual"><geometry><sphere><radius>{r}</radius></sphere></geometry><material><ambient>0.5 0.5 0.5 1</ambient><diffuse>0.5 0.5 0.5 1</diffuse></material></visual></link></model>'''
            library=ROOT/'build/libra_obstacle_motion.so'
            if not library.exists():raise RuntimeError('Obstacle plugin not built')
            model+=f'''<plugin filename="{library}" name="RaObstacleMotion"><data_file>{datafile}</data_file><offset>{off[0]} {off[1]} {off[2]}</offset></plugin>'''
        src=src.replace('</world>',model+'</world>');world.write_text(src,encoding='utf8')
        (self.out/'world.sdf').write_text(src,encoding='utf8')
        env=os.environ.copy();env['PATH']=str(Path(PY).parent)+':/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin'
        for k in ('CMAKE_PREFIX_PATH','CMAKE_MODULE_PATH','CPATH','CPLUS_INCLUDE_PATH','C_INCLUDE_PATH','LIBRARY_PATH','LD_LIBRARY_PATH','PYTHONPATH','PYTHONHOME','CONDA_PREFIX','CONDA_DEFAULT_ENV'):env.pop(k,None)
        env.update(PX4_SIM_SPEED_FACTOR='1',PX4_GZ_WORLD='ra_lab',PX4_GZ_MODEL_NAME='x500_0')
        if not self.gui:env['HEADLESS']='1'
        else:env.pop('HEADLESS',None)
        # gz_env.sh expands this build variable; use a process-local symlinked world path.
        link=PX4/'Tools/simulation/gz/worlds/ra_lab.sdf'
        if link.is_symlink():
            previous=str(link.readlink())
            if 'ra_closed_loop_20261030' not in previous and 'continuation_20261101' not in previous and 'live_deployment_20261101' not in previous and 'shared_visuals_20261101' not in previous and 'continuous_execution_20261101' not in previous and 'dense_validation_20261101' not in previous and 'dense_execution_fix_20261102' not in previous and 'dense_tracking_20261102' not in previous and 'first_arrival_comparison_20261102' not in previous and 'paper_execution_evidence_20261102' not in previous and 'tight_bound_candidate_qp_20261102' not in previous and 'directional_execution_20261102' not in previous and 'bounded_runtime_20261102' not in previous and 'tracking_fair_comparison_20261102' not in previous and 'aligned_state_comparison_20261102' not in previous and 'verified_execution_20261102' not in previous and 'verified_tracking_20261103' not in previous and 'paper_comparison_20261103' not in previous and 'execution_pipeline_20261103' not in previous and 'finite_execution_20261103' not in previous and 'candidate_runtime_20261103' not in previous and 'git_execution_native_20261103' not in previous and 'git_execution_feedback_20261103' not in previous and 'git_execution_attitude_20261103' not in previous and 'git_execution_auditfix_20261103' not in previous and 'git_execution_timingfix_20261104' not in previous and 'git_scene_variation_20261105' not in previous:raise RuntimeError('Unowned world symlink preserved')
            link.unlink()
        if link.exists() and not link.is_symlink():raise RuntimeError('Unowned world path preserved')
        if not link.exists():link.symlink_to(world)
        self.logger_config=PX4/'build/px4_sitl_default/rootfs/etc/logging/logger_topics.txt'
        self.logger_old=self.logger_config.read_bytes() if self.logger_config.exists() else None
        topics=['trajectory_setpoint 0','vehicle_local_position_groundtruth 4','vehicle_attitude_groundtruth 4','vehicle_local_position 10','vehicle_local_position_setpoint 4','actuator_motors 4','vehicle_acceleration 4','vehicle_attitude 4','vehicle_status 20','vehicle_land_detected 20','vehicle_command_ack 0','timesync_status 20','logger_status 100','vehicle_angular_velocity 4','vehicle_rates_setpoint 4','vehicle_attitude_setpoint 4','offboard_control_mode 0','estimator_status_flags 20','vehicle_global_position 100']
        self.logger_bytes=bytes(('\n'.join(topics)+'\n'),'ascii')
        self.logger_config.parent.mkdir(parents=True,exist_ok=True);self.logger_config.write_bytes(self.logger_bytes)
        (self.out/'logger_topics.txt').write_bytes(self.logger_bytes)
        self.log=(self.out/'px4_gazebo.log').open('wb');self.started=time.time()
        env['PX4_SIM_MODEL']='gz_x500'
        env['PATH']=str(PX4/'build/px4_sitl_default/bin')+':'+env['PATH']
        self.proc=subprocess.Popen([str(FIRMWARE),str(PX4/'build/px4_sitl_default/etc'),'-w',str(PX4/'build/px4_sitl_default/rootfs')],cwd=PX4,env=env,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(12)
        if self.proc.poll() is not None:raise RuntimeError('PX4 exited at startup: '+str(self.out))
        return self
    def __exit__(self,*exc):
        if self.proc is not None:
            # The mission is already physically paused; drain the actual logger
            # without advancing Gazebo, sending control or changing arrival time.
            flush=dict(requested=False,error=None,world_frozen_before_flush=False)
            summary=self.out/'summary.json'
            if summary.exists():flush['world_frozen_before_flush']=json.loads(summary.read_text()).get('post_terminal_world_paused',False)
            if flush['world_frozen_before_flush']:
                try:
                    self.proc.stdin.write(b'logger stop\n');self.proc.stdin.flush();flush['requested']=True;time.sleep(.25)
                except (OSError,BrokenPipeError) as err:flush['error']=repr(err)
            (self.out/'logger_flush.json').write_text(json.dumps(flush,indent=2))
            try:os.killpg(self.proc.pid,signal.SIGINT)
            except ProcessLookupError:pass
            try:self.proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid,signal.SIGTERM)
                try:self.proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    # Only our own new session, after physical termination and logger flush.
                    assert os.getpgid(self.proc.pid)==self.proc.pid
                    os.killpg(self.proc.pid,signal.SIGKILL);self.proc.wait(timeout=5)
        from scoped_cleanup import stop_owned_group
        stop_owned_group(self.proc.pid,self.out)
        self.log.close()
        if self.logger_config.exists() and self.logger_config.read_bytes()==self.logger_bytes:
            if self.logger_old is None:self.logger_config.unlink()
            else:self.logger_config.write_bytes(self.logger_old)
        ulgs=[p for p in (PX4/'build/px4_sitl_default/rootfs/log').rglob('*.ulg') if p.stat().st_mtime>=self.started-2]
        if ulgs:
            p=max(ulgs,key=lambda a:a.stat().st_mtime)
            with p.open('rb') as inp,gzip.open(self.out/'flight.ulg.gz','wb',compresslevel=5) as dest:shutil.copyfileobj(inp,dest)
        receipt=dict(SITL_only=True,real_aircraft_connected=False,px4_returncode=self.proc.returncode,
            firmware_commit=subprocess.check_output(['git','-C',str(PX4),'rev-parse','HEAD'],text=True).strip(),
            firmware_sha256=hashlib.sha256(FIRMWARE.read_bytes()).hexdigest(),
            exception=str(exc[1]) if exc[1] else None,
            files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in self.out.iterdir() if p.is_file()})
        (self.out/'run_receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf8')
