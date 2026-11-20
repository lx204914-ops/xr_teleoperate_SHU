"""Computation warm-up and bounded mission-thread scheduling; same equations."""
from pathlib import Path
import numpy as np
import os,time,json
INSTALLED=False
ENABLED=True
def timed(function,key):
 def call(*args,**kwargs):
  if not ENABLED:return function(*args,**kwargs)
  from runtime_optimizer import STATS
  start=time.perf_counter()
  try:return function(*args,**kwargs)
  finally:STATS[key]=STATS.get(key,0.)+(time.perf_counter()-start)*1000
 return call
def install():
 global INSTALLED
 if INSTALLED:return
 import px4_adapter_base as base,conditional_gate
 base.interval_margin_rows=timed(base.interval_margin_rows,'margin_ms')
 base.constraints=timed(base.constraints,'coefficients_ms')
 base.DeployedController.solve_nominal=timed(base.DeployedController.solve_nominal,'nominal_ms')
 conditional_gate.current_common_set=timed(conditional_gate.current_common_set,'common_set_ms')
 INSTALLED=True
def warm_common_solver(n):
 from ra_dpcbf.certificates import current_common_set
 # Same four-variable LP transport and actual row count, disposable problem.
 B=np.r_[np.tile([[1.,0.,0.]],(n,1)),np.eye(3),-np.eye(3),[[0.,0.,1.]]]
 C=np.r_[np.full(n,10.),np.full(6,1.5),10.]
 for _ in range(4):assert current_common_set(B,C,np.full(3,2.))['feasible']
def request_scheduling(out):
 out=Path(out);root=out.parents[1]
 request=dict(pid=os.getpid(),uid=os.getuid(),session=out.name,script=str(root/'code/run_continuous.py'),physics_not_started=False,mission_not_started=True)
 (out/'control_scheduling_request.json').write_text(json.dumps(request,indent=2))
 selected=int(json.loads((root/'qa/control_cpu_benchmark.json').read_text())['selected_core'])
 assert selected in os.sched_getaffinity(0),'Selected CPU is not available'
 from affinity_isolation77 import isolate
 isolate(out,original_allowed=set(os.sched_getaffinity(0)))
 os.sched_setaffinity(0,{selected})
 policy=os.sched_getscheduler(0)
 receipt=dict(passed=os.sched_getaffinity(0)=={selected},pid=os.getpid(),session=out.name,affinity=sorted(os.sched_getaffinity(0)),policy=policy,priority=os.sched_getparam(0).sched_priority,scope='Unprivileged exact-session affinity; own background threads excluded from control physical siblings; no root helper or persistent scheduled task',hard_realtime_guarantee=False)
 (out/'control_scheduling_receipt.json').write_text(json.dumps(receipt,indent=2))
 assert receipt['passed'],receipt
