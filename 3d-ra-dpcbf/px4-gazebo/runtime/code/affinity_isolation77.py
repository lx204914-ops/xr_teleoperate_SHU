"""Unprivileged affinity for this exact SITL partition, before the mission.

Only owned same-UID processes with the exact random GZ_PARTITION are touched.
No root task, global CPU isolation, priority escalation, or algorithm change.
"""
from pathlib import Path
import os,json,time

def parse_mask(text):
    values=set()
    for p in text.strip().split(','):
        a,b=map(int,p.split('-')) if '-' in p else (int(p),int(p))
        values.update(range(a,b+1))
    return values

def isolate(out,original_allowed=None):
    out=Path(out);partition=os.environ.get('GZ_PARTITION','')
    assert partition.startswith('cbf_compare_'+out.name), 'Refuse unknown process scope'
    state=out/'cpu_isolation77.json';prior=json.loads(state.read_text()) if state.exists() else {}
    allowed=set(original_allowed or prior.get('original_allowed',[]));assert allowed
    root=out.parents[1];core=int(json.loads((root/'qa/control_cpu_benchmark.json').read_text())['selected_core'])
    sibling=parse_mask(Path(f'/sys/devices/system/cpu/cpu{core}/topology/thread_siblings_list').read_text())
    others=allowed-sibling;assert others and core in allowed
    changes=[];skipped_unreadable=[];uid=os.getuid();me=os.getpid()
    for p in Path('/proc').iterdir():
        if not p.name.isdecimal():continue
        try:
            if p.stat().st_uid!=uid:continue
            try:raw=(p/'environ').read_bytes()
            except PermissionError:
                skipped_unreadable.append(int(p.name));continue
            env=dict(x.split(b'=',1) for x in raw.split(b'\0') if b'=' in x)
            if env.get(b'GZ_PARTITION',b'').decode()!=partition:continue
            for thread in (p/'task').iterdir():
                tid=int(thread.name)
                if tid==me:continue
                before=sorted(os.sched_getaffinity(tid));os.sched_setaffinity(tid,others)
                after=sorted(os.sched_getaffinity(tid));assert not (set(after)&sibling)
                changes.append(dict(pid=int(p.name),tid=tid,before=before,after=after))
        except (ProcessLookupError,FileNotFoundError):continue
    os.sched_setaffinity(0,{core})
    proof=dict(passed=bool(changes),partition=partition,uid=uid,control_pid=me,control_affinity=sorted(os.sched_getaffinity(0)),excluded_siblings=sorted(sibling),original_allowed=sorted(allowed),other_affinity=sorted(others),changed_threads=changes,unreadable_unscoped_PIDs_skipped=skipped_unreadable,stage='before_mission',updated_unix=time.time(),prior_passes=prior.get('prior_passes',0)+1,unprivileged=True,hard_realtime_guarantee=False,no_unrelated_processes_touched=True)
    assert proof['passed'];state.write_text(json.dumps(proof,indent=2));return proof
