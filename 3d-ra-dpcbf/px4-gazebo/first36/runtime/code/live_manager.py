"""Start/stop one real PX4 SITL flight, retain native GUI, never replay positions."""
from pathlib import Path
import sys,os,json,subprocess,time,signal,argparse,uuid
ROOT=Path(__file__).resolve().parents[1];PY='/home/xueyuan/cbf_px4/.venv/bin/python'
os.environ['GZ_IP']='127.0.0.1'

def alive(pid):
    try:os.kill(pid,0);return True
    except (ProcessLookupError,PermissionError):return False

def read_status():
    p=ROOT/'active_session.json'
    return json.loads(p.read_text()) if p.exists() else {}

def save(s):
    p=ROOT/'active_session.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(s,indent=2));tmp.replace(p)

def worker(name,preset):
    (ROOT/'runs').mkdir(exist_ok=True);(ROOT/'qa').mkdir(exist_ok=True)
    state=dict(session=name,worker_pid=os.getpid(),phase='starting_real_SITL',completed=False,preset=preset)
    save(state);env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',GZ_IP='127.0.0.1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    previous=ROOT/'native_gui.json';pids=[]
    if previous.exists():pids.append(json.loads(previous.read_text()).get('pid'))
    if os.environ.get('CBF_REPLACE_OLD_GUI')=='6066':pids.append(6066)
    for pid in pids:
        if not pid:continue
        p=Path('/proc')/str(pid)/'cmdline'
        if p.exists() and p.read_bytes().startswith(b'gz sim -g'):os.kill(pid,signal.SIGTERM)
        until=time.monotonic()+3.
        while p.exists() and time.monotonic()<until:
            status=p.parent/'stat'
            if status.exists() and status.read_text().split(') ')[-1].startswith('Z'):break
            time.sleep(.1)
    # The upstream PX4 startup checks topic NAMES, including subscriptions.
    # A leftover GUI/clock subscriber can make it incorrectly assume a server
    # exists. Close only our clients first; don't subscribe until scene service.
    time.sleep(.8)
    if (ROOT/'STOP_AFTER_LANDING').exists():(ROOT/'STOP_AFTER_LANDING').unlink()
    log=(ROOT/'runs'/(name+'.log')).open('wb')
    flight=subprocess.Popen([PY,str(ROOT/'code/run_continuous.py'),'--name',name,'--test','ten','--scene',str(ROOT/'scenes'/('ten_observe28.npz' if preset=='frozen' else 'ten_faster_same_layout.npz')),'--method','ra','--duration',('28' if preset=='frozen' else '32'),'--execution','velocity_ff','--native-view','--wall-rtf','1.0','--cruise',('1.0' if preset=='frozen' else '1.5')],stdout=log,stderr=subprocess.STDOUT,env=env)
    state['flight_pid']=flight.pid;save(state)
    begin=time.monotonic()
    while flight.poll() is None and time.monotonic()-begin<45:
        providers=subprocess.run(['gz','service','-i','-s','/world/ra_lab/scene/info'],capture_output=True,text=True,env=env,timeout=5)
        if 'Service providers' in providers.stdout:break
        time.sleep(.3)
    gui_env=dict(QSG_RENDER_LOOP='basic',GZ_GUI_PLUGIN_PATH=str(ROOT/'build/native_renderer'),PATH='/usr/bin:/bin',HOME='/home/xueyuan',DISPLAY=':1',XAUTHORITY='/run/user/1000/gdm/Xauthority',QT_QPA_PLATFORM='xcb',__GLX_VENDOR_LIBRARY_NAME='nvidia',GZ_IP='127.0.0.1',XDG_RUNTIME_DIR='/run/user/1000',GZ_SIM_RESOURCE_PATH='/home/xueyuan/cbf_px4/PX4-Autopilot/Tools/simulation/gz/models')
    attempts=[]
    for attempt in range(1):
        # Prefer real 5090 Ogre rendering. Only fall back if both hardware
        # context attempts exit; never label llvmpipe as GPU acceleration.
        if attempt==2:gui_env.update(GALLIUM_DRIVER='llvmpipe',LIBGL_ALWAYS_SOFTWARE='1',LP_NUM_THREADS='4')
        with (ROOT/'native_gui.log').open('ab') as f:gui=subprocess.Popen(['gz','sim','-g','-v','2','--gui-config',str(ROOT/'code/native_gui.config')],env=gui_env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        from native_window import configure
        wid=configure(gui.pid,gui_env['DISPLAY'])
        info=dict(pid=gui.pid,display=':1',window_id=wid,native=True,session=name,renderer='native_NVIDIA')
        tmp=ROOT/'native_gui.tmp';tmp.write_text(json.dumps(info));tmp.replace(ROOT/'native_gui.json')
        time.sleep(.2)
        attempts.append(dict(attempt=attempt+1,pid=gui.pid,backend='native_NVIDIA',exited=gui.poll() is not None))
        if gui.poll() is None:break
        time.sleep(1.)
    state['gui_attempts']=attempts;save(state)
    state.update(phase='live_physics_and_PX4',gui_pid=gui.pid);save(state)
    rc=flight.wait();log.close();state.update(flight_returncode=rc,phase='auditing_saved_physics')
    save(state);folder=ROOT/'runs'/name
    if (folder/'summary.json').exists():
        with (folder/'audit.log').open('wb') as f:
            audit=subprocess.run([PY,str(ROOT/'code/verify_avoidance.py'),str(folder)],stdout=f,stderr=subprocess.STDOUT,timeout=120)
        state['summary']=json.loads((folder/'summary.json').read_text())
        if (folder/'independent_check.json').exists():state['audit']=json.loads((folder/'independent_check.json').read_text())
    if (folder/'summary.json').exists() and (folder/'execution_trace.json').exists():
        with (folder/'continuous_audit.log').open('wb') as f:
            result=subprocess.run([PY,str(ROOT/'code/audit_continuous.py'),str(folder)],stdout=f,stderr=subprocess.STDOUT,timeout=120)
        state['continuous_audit_returncode']=result.returncode
        if (folder/'continuous_check.json').exists():state['continuous_audit']=json.loads((folder/'continuous_check.json').read_text())
    if (folder/'display_snapshots.npz').exists() and (folder/'native_frame_audit.npz').exists():
        with (folder/'display_audit.log').open('wb') as f:
            result=subprocess.run([PY,str(ROOT/'code/audit_relative_display.py'),str(folder)],stdout=f,stderr=subprocess.STDOUT,timeout=120)
        state['display_audit_returncode']=result.returncode
        if (folder/'relative_display_check.json').exists():state['display_audit']=json.loads((folder/'relative_display_check.json').read_text())
    state.update(completed=True,phase='flight_finished_native_GUI_retained');save(state)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('start','stop','status','worker'));ap.add_argument('--name');ap.add_argument('--preset',choices=('frozen','faster'),default='faster');args=ap.parse_args()
    if args.action=='status':print(json.dumps(read_status(),indent=2));return
    if args.action=='stop':(ROOT/'STOP_AFTER_LANDING').touch();print('Stop requested; runner will land normally.');return
    if args.action=='worker':worker(args.name,args.preset);return
    old=read_status()
    if old and not old.get('completed') and alive(old.get('worker_pid',0)):raise RuntimeError('Real flight already active; no duplicate start')
    if subprocess.run(['pgrep','-x','px4'],capture_output=True).returncode==0:raise RuntimeError('Existing PX4 belongs to another session; preserved')
    name='continuous_ten_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]
    with (ROOT/'manager.log').open('ab') as f:
        p=subprocess.Popen([PY,str(Path(__file__).resolve()),'worker','--name',name,'--preset',args.preset],stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    print(json.dumps(dict(started_real_flight=True,session=name,worker_pid=p.pid)))

if __name__=='__main__':main()
