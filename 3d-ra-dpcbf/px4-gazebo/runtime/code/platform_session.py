"""Own and stop only one PX4/Gazebo SITL process group; no aircraft interfaces."""
from pathlib import Path
import os,subprocess,time,signal,gzip,shutil,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
PX4=Path(os.environ.get('CBF_PX4_ROOT',str(Path.home()/'cbf_px4/PX4-Autopilot')))
PY=os.environ.get('CBF_PYTHON',str(Path.home()/'cbf_px4/.venv/bin/python'))
FIRMWARE=Path(os.environ.get('CBF_PX4_FIRMWARE',str(ROOT/'build/feedback_receipt_bin/px4')))
class Platform:
    def __init__(self,out,gui=False,motion=None,tape_dt=None,initial_offset=None):
        self.out=Path(out).expanduser().resolve();self.gui=gui;self.proc=None;self.motion=motion;self.tape_dt=tape_dt;self.initial_offset=initial_offset
        self.log=None;self._link=None;self._link_old=None;self._installed_link=None;self._logger_path=None;self._logger_snapshot=None
    def _preflight(self):
        if self.out.exists():raise FileExistsError('Existing output preserved: '+str(self.out))
        required=[FIRMWARE,Path(PY),PX4/'Tools/simulation/gz/worlds/default.sdf',PX4/'src/modules/simulation/gz_bridge/server.config',ROOT/'build/libra_obstacle_motion.so',ROOT/'build/libperiodic_publish75.so']
        for p in required:
            if not p.is_file():raise FileNotFoundError('Required platform input missing: '+str(p))
        if os.name=='posix' and not os.access(FIRMWARE,os.X_OK):raise PermissionError('Firmware is not executable: '+str(FIRMWARE))
        for command in ('pgrep','gz','git'):
            if shutil.which(command) is None:raise FileNotFoundError('Required command missing: '+command)
        import xml.etree.ElementTree as ET
        ET.parse(required[2]);ET.parse(required[3])
        self._link=PX4/'Tools/simulation/gz/worlds/ra_lab.sdf'
        if self._link.is_symlink():
            target=self._link.resolve()
            if target!=(ROOT/'worlds/ra_lab.sdf').resolve() and not target.is_relative_to((ROOT/'runs').resolve()):
                raise RuntimeError('Unowned world symlink preserved: '+str(self._link))
            self._link_old=str(self._link.readlink())
        elif self._link.exists():raise RuntimeError('Unowned world path preserved: '+str(self._link))
        logger=PX4/'build/px4_sitl_default/rootfs/etc/logging/logger_topics.txt'
        if logger.is_symlink():raise RuntimeError('Logger symlink preserved: '+str(logger))
        self._logger_path=logger
        self._logger_snapshot=logger.read_bytes() if logger.exists() else None
        self._logger_mode=(logger.stat().st_mode & 0o777) if logger.exists() else 0o644

    def _atomic_logger_write(self,data):
        import tempfile
        p=self._logger_path;p.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='.cbf-logger-',dir=p.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(data)
            os.chmod(name,self._logger_mode)
            os.replace(name,p)
        finally:Path(name).unlink(missing_ok=True)

    def _install_world_link(self,world):
        if self._link.is_symlink():
            if str(self._link.readlink())!=self._link_old:raise RuntimeError('World link changed during preparation; preserved')
            self._link.unlink()
        elif self._link.exists():raise RuntimeError('World path changed during preparation; preserved')
        self._installed_link=str(world)
        self._link.symlink_to(world)

    def _stop_started_process(self):
        if self.proc is None:return
        # Popen below always uses start_new_session=True: this is our private group.
        try:os.killpg(self.proc.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        try:self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:os.killpg(self.proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            self.proc.wait(timeout=5)
        # A launcher may exit before its Gazebo child. Reap only the same group.
        try:
            os.killpg(self.proc.pid,0)
            os.killpg(self.proc.pid,signal.SIGKILL)
        except ProcessLookupError:pass

    def _restore_local_state(self):
        errors=[]
        try:
            if self.log is not None and not self.log.closed:self.log.close()
        except OSError as err:errors.append(repr(err))
        p=self._logger_path
        try:
            if p is not None:
                data=p.read_bytes() if p.exists() else None
                expected=getattr(self,'logger_bytes',None)
                if data==self._logger_snapshot:pass
                elif expected is not None and data==expected:
                    if self._logger_snapshot is None:p.unlink()
                    else:self._atomic_logger_write(self._logger_snapshot)
                else:errors.append('Logger changed by another writer; preserved')
        except OSError as err:errors.append(repr(err))
        try:
            if self._link is not None and self._installed_link is not None:
                if self._link.is_symlink() and str(self._link.readlink())==self._installed_link:self._link.unlink()
                elif self._link.exists() or self._link.is_symlink():
                    errors.append('World link changed by another writer; preserved')
                if not self._link.exists() and not self._link.is_symlink() and self._link_old is not None:self._link.symlink_to(self._link_old)
        except OSError as err:errors.append(repr(err))
        if errors:raise RuntimeError('Platform restoration incomplete: '+ '; '.join(errors))

    def __enter__(self):
        self._preflight()
        try:return self._start()
        except BaseException as original:
            try:
                try:self._stop_started_process()
                finally:self._restore_local_state()
            except Exception as cleanup:
                if hasattr(original,'add_note'):original.add_note(str(cleanup))
            raise

    def __exit__(self,*exc):
        try:return self._finish(*exc)
        finally:
            try:self._stop_started_process()
            finally:self._restore_local_state()

    def _start(self):
        if subprocess.run(['pgrep','-x','px4'],capture_output=True).returncode==0:raise RuntimeError('Existing PX4 preserved')
        self.out.mkdir(parents=True,exist_ok=False)
        if (ROOT/'stage_lock.json').exists():(self.out/'stage_lock_at_run.json').write_bytes((ROOT/'stage_lock.json').read_bytes())
        if (ROOT/'execution_contract.json').exists():(self.out/'execution_contract.json').write_bytes((ROOT/'execution_contract.json').read_bytes())
        world=self.out/'world.sdf'
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
        self._install_world_link(world)
        self.logger_config=PX4/'build/px4_sitl_default/rootfs/etc/logging/logger_topics.txt'
        self.logger_old=self.logger_config.read_bytes() if self.logger_config.exists() else None
        topics=['trajectory_setpoint 0','vehicle_local_position_groundtruth 4','vehicle_attitude_groundtruth 4','vehicle_local_position 10','vehicle_local_position_setpoint 4','actuator_motors 4','vehicle_acceleration 4','vehicle_attitude 4','vehicle_status 20','vehicle_land_detected 20','vehicle_command_ack 0','timesync_status 20','logger_status 100','vehicle_angular_velocity 4','vehicle_rates_setpoint 4','vehicle_attitude_setpoint 4','offboard_control_mode 0','estimator_status_flags 20','vehicle_global_position 100']
        self.logger_bytes=bytes(('\n'.join(topics)+'\n'),'ascii')
        if self.logger_config.is_symlink() or self.logger_old!=self._logger_snapshot:raise RuntimeError('Logger changed during preparation; preserved')
        self._atomic_logger_write(self.logger_bytes)
        (self.out/'logger_topics.txt').write_bytes(self.logger_bytes)
        self.log=(self.out/'px4_gazebo.log').open('wb');self.started=time.time()
        env['PX4_SIM_MODEL']='gz_x500'
        env['PATH']=str(PX4/'build/px4_sitl_default/bin')+':'+env['PATH']
        self.proc=subprocess.Popen([str(FIRMWARE),str(PX4/'build/px4_sitl_default/etc'),'-w',str(PX4/'build/px4_sitl_default/rootfs')],cwd=PX4,env=env,stdin=subprocess.PIPE,stdout=self.log,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(12)
        if self.proc.poll() is not None:raise RuntimeError('PX4 exited at startup: '+str(self.out))
        return self
    def _finish(self,*exc):
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
            else:self._atomic_logger_write(self.logger_old)
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
