"""Bounded flight timing, CPU/scheduler attribution, and deferred cyclic GC."""
from pathlib import Path
import time,os,gc,resource,json
from runtime_optimizer import stats

class PerformanceMonitor:
    def __init__(self):self.rows=[];self.gc_enabled=gc.isenabled();self.started=False
    def start_mission(self):
        # Reference-count reclamation remains active. This bounded flight defers
        # only cyclic collection; restore and collect after landing, not in QP.
        gc.collect();gc.disable();self.started=True
    def begin(self,t):
        self.t=float(t);self.cpu=time.thread_time_ns();self.before=stats();self.scheduler=None;self.use=resource.getrusage(resource.RUSAGE_THREAD)
    def end(self,wall,diagnostics,result=None):
        cpu=(time.thread_time_ns()-self.cpu)*1e-6;after=stats();sched=None;use=resource.getrusage(resource.RUSAGE_THREAD)
        r=dict(t=self.t,wall_ms=float(wall*1000),cpu_ms=cpu,runqueue_wait_ms=None,
            voluntary_switches=use.ru_nvcsw-self.use.ru_nvcsw,involuntary_switches=use.ru_nivcsw-self.use.ru_nivcsw,
            support_queries=after['queries']-self.before['queries'],support_direct_LPs=after['direct_lps']-self.before['direct_lps'],
            support_vertex_LPs=after['vertex_lps']-self.before['vertex_lps'],support_vertex_hits=after['vertex_hits']-self.before['vertex_hits'])
        r.update(kkt_certificate_ms=after.get('kkt_certificate_ms',0)-self.before.get('kkt_certificate_ms',0),projection_solver_ms=after.get('projection_solver_ms',0)-self.before.get('projection_solver_ms',0))
        if diagnostics is not None:
            d=diagnostics.get('detour',{});r.update(active=bool(d.get('active',False)),event=int(d.get('event',0)),reason=d.get('reason','not_applicable_to_baseline'))
        if result is not None:
            r.update(direction_ms=float(result.direction_seconds*1000),qp_ms=float(result.qp_seconds*1000))
        for name in ('margin_ms','nominal_ms','coefficients_ms','common_set_ms','candidate_qp_ms','candidate_certificate_ms'):
            r[name]=after.get(name,0.)-self.before.get(name,0.)
        self.rows.append(r)
    def save(self,out):
        if self.gc_enabled:gc.enable()
        start=time.perf_counter();collected=gc.collect();gc_ms=(time.perf_counter()-start)*1000
        data=dict(rows=self.rows,backend_totals=stats(),deferred_cyclic_collection=dict(enabled=self.started,restored=gc.isenabled(),after_landing_ms=gc_ms,objects_collected=collected),
            maximum_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.,note='Main-thread CPU excludes certificate worker CPU; wall time includes waiting for every worker. All timing includes instrumentation cost; simulation clock is separately verified.')
        (Path(out)/'performance_breakdown.json').write_text(json.dumps(data,indent=2))
