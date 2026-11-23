"""Serial, frozen-scene comparisons with immediate virtual termination at arrival."""
from pathlib import Path
import sys,os,json,subprocess,time,argparse,uuid,signal,hashlib
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT.parent;PY=str(BASE/'.venv/bin/python')
LABELS={'ra':'RA-DPCBF','ma_cbf_vo':'MA-CBF-VO','c3bf':'C3BF (fixed direction cone schematic)','park':'Park-inspired DPCBF'}
def load(path,default=None):
 try:return json.loads(path.read_text())
 except (OSError,ValueError):return {} if default is None else default
def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)
def close_owned_gui():
 for directory in ('dense_tracking_20261102','continuous_execution_20261101','first_arrival_comparison_20261102','paper_execution_evidence_20261102','tight_bound_candidate_qp_20261102','directional_execution_20261102','bounded_runtime_20261102','tracking_fair_comparison_20261102','aligned_state_comparison_20261102','verified_execution_20261102','verified_tracking_20261103','paper_comparison_20261103','execution_pipeline_20261103','finite_execution_20261103','candidate_runtime_20261103','git_execution_native_20261103','git_execution_feedback_20261103','git_execution_attitude_20261103'):
  pid=load(BASE/directory/'native_gui.json').get('pid')
  if not pid:continue
  p=Path('/proc')/str(pid)/'cmdline'
  try:cmd=p.read_bytes()
  except OSError:continue
  if ' '.join(cmd.replace(b'\0',b' ').decode().split()).startswith('gz sim -g ') and ('--gui-config '+str(BASE/directory/'code/native_gui.config')) in ' '.join(cmd.replace(b'\0',b' ').decode().split()):
   try:os.kill(pid,signal.SIGTERM)
   except ProcessLookupError:pass
 time.sleep(.8)
def worker(name,case,method):
 config=load(ROOT/'comparison_contract.json')['cases'][case]
 os.environ.update(GZ_IP='127.0.0.1',GZ_PARTITION='cbf_compare_'+name,CBF_COMPARE_METHOD=method,CBF_METHOD_LABEL=LABELS[method],HOME='/home/xueyuan',XAUTHORITY='/run/user/1000/gdm/Xauthority',XDG_RUNTIME_DIR='/home/xueyuan/.cache/cbf-runtime',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',NUMBA_NUM_THREADS='1',PYTHONUNBUFFERED='1')
 close_owned_gui()
 command=[PY,str(ROOT/'code/run_continuous.py'),'--name',name,'--test',case,'--scene',config['scene'],'--method',method,'--duration',str(config['duration']),'--execution','acceleration','--native-view','--wall-rtf','1.0','--cruise','1.5']
 if config['initialize']:command.append('--initialize')
 source=(ROOT/'code/live_manager.py').read_text().replace('/home/xueyuan','/home/xueyuan').replace("XDG_RUNTIME_DIR='/run/user/1000'","XDG_RUNTIME_DIR='/home/xueyuan/.cache/cbf-runtime'")
 line=next(v for v in source.splitlines() if v.startswith('    flight=subprocess.Popen('))
 source=source.replace(line,"    flight=subprocess.Popen(COMPARE_COMMAND,stdout=log,stderr=subprocess.STDOUT,env=env)")
 wait="    attach_deadline=time.monotonic()+100\n    while flight.poll() is None and not (ROOT/'runs'/name/'native_gui_attach_ready').exists() and time.monotonic()<attach_deadline:time.sleep(.1)\n    if not (ROOT/'runs'/name/'native_gui_attach_ready').exists():raise RuntimeError('Physical preparation did not complete before GUI attachment')\n"
 source=source.replace('    gui_env=dict(',wait+'    gui_env=dict(GZ_PARTITION=os.environ[\'GZ_PARTITION\'],CBF_METHOD_LABEL=os.environ[\'CBF_METHOD_LABEL\'],')
 source=source.replace('    attempts=[]\n',"    gui_env.update(__GLX_VENDOR_LIBRARY_NAME='nvidia')\n    attempts=[]\n")
 source=source.replace("str(ROOT/'code/audit_relative_display.py')","str(ROOT/'code/audit_comparison_display.py')")
 source=source.replace("if (folder/'display_snapshots.npz').exists() and (folder/'native_frame_audit.npz').exists():","if (folder/'native_view_status.json').exists():")
 scope=dict(__file__=str(ROOT/'code/live_manager.py'),__name__='_comparison_lifecycle',COMPARE_COMMAND=command)
 exec(compile(source,str(ROOT/'code/live_manager.py'),'exec'),scope)
 scope['worker'](name,case)
def start(case,method):
 assert subprocess.run(['pgrep','-x','px4'],capture_output=True).returncode!=0,'Existing real PX4 preserved'
 for key,h in load(ROOT/'source_lock.json').items():assert hashlib.sha256((ROOT/key.replace(chr(92),'/')).read_bytes()).hexdigest()==h,key
 name='compare_'+case+'_'+method+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:4]
 env={**os.environ,'GZ_PARTITION':'cbf_compare_'+name,'CBF_COMPARE_METHOD':method,'CBF_METHOD_LABEL':LABELS[method],'GZ_IP':'127.0.0.1'}
 with (ROOT/'manager.log').open('ab') as log:
  w=subprocess.Popen([PY,str(Path(__file__).resolve()),'worker','--name',name,'--scene',case,'--method',method],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 with (ROOT/('record_'+name+'.log')).open('wb') as log:
  c=subprocess.Popen([PY,str(ROOT/'code/record_comparison.py'),'--attach-session',name],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 result=dict(session=name,worker_pid=w.pid,recorder_pid=c.pid,case=case,method=method)
 save(ROOT/'latest_launch.json',result);return result
def queue():
 import fcntl
 guard=(ROOT/'queue.lock').open('w');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 contract=load(ROOT/'comparison_contract.json');status=load(ROOT/'queue_status.json',dict(running=True,completed=[],total=16));status['running']=True
 for case in contract['cases']:
  for method in contract['methods']:
   if any(v['case']==case and v['method']==method for v in status['completed']):continue
   if (ROOT/'STOP_QUEUE').exists():status.update(running=False,phase='user_stopped');save(ROOT/'queue_status.json',status);return
   item=start(case,method);status.update(current=item,phase='real_physical_run');save(ROOT/'queue_status.json',status)
   print('REAL_COMPARISON_STARTED',json.dumps(item),flush=True)
   deadline=time.monotonic()+310
   while time.monotonic()<deadline:
    s=load(ROOT/'active_session.json');rec=load(ROOT/'native_recording'/item['session']/'recording_status.json')
    if s.get('session')==item['session'] and s.get('completed') and rec.get('phase') in ('completed_real_flight_and_GIF','completed_short_failure_native_PNG','completed_without_recording'):
     break
    process=Path('/proc')/str(item['worker_pid'])/'stat'
    dead=not process.exists()
    if not dead:
     try:dead=process.read_text().split(') ')[-1].startswith('Z')
     except OSError:dead=True
    if dead and (ROOT/'runs'/item['session']/'summary.json').exists() and (s.get('completed') or rec.get('phase','').startswith('completed')):break
    if dead and not (ROOT/'runs'/item['session']/'summary.json').exists():
     status.update(running=False,phase='worker_setup_error',error='Read manager.log; no mission summary');save(ROOT/'queue_status.json',status);raise RuntimeError(status['error'])
    time.sleep(.4)
   else:
    status.update(running=False,phase='bounded_wait_expired_no_automatic_retry');save(ROOT/'queue_status.json',status);raise RuntimeError(status['phase'])
   folder=ROOT/'runs'/item['session'];summary=load(folder/'summary.json');a=load(folder/'independent_check.json');c=load(folder/'continuous_check.json')
   result={**item,'completed':summary.get('completed',False),'failure':summary.get('failure'),'arrival_s':a.get('first_arrival_truth_s'),'physical_passed':a.get('passed',False),'execution_passed':c.get('passed',False)}
   status['completed'].append(result);status.update(current=None,phase='archiving');save(ROOT/'queue_status.json',status)
   subprocess.run([PY,str(ROOT/'code/build_comparison_report.py')],check=True,timeout=60)
 status.update(running=False,phase='all_16_finished',current=None);save(ROOT/'queue_status.json',status)
 subprocess.run([PY,str(ROOT/'code/build_comparison_report.py')],check=True,timeout=60)
 print('ALL_COMPARISONS_FINISHED',flush=True)
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('action',choices=('start','worker','queue'));ap.add_argument('--case',type=int,choices=(50,100));ap.add_argument('--scene',choices=('four','ten','fifty','hundred'));ap.add_argument('--method',choices=tuple(LABELS),default='ra');ap.add_argument('--name');args=ap.parse_args()
 if args.action=='worker':worker(args.name,args.scene,args.method)
 elif args.action=='queue':queue()
 else:print(json.dumps(start(args.scene or ('fifty' if args.case==50 else 'hundred'),args.method)),flush=True)
