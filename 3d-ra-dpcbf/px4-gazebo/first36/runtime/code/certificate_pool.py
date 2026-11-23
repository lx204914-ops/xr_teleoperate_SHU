"""Parallel independent obstacle certificates; original scalar arithmetic/order per row."""
import os,time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from candidate_hold_bitround75 import candidate_lower as scalar_lower
_pool=None

def _compute(x,obs,par,u,dt,delta,segments):
 global _pool
 if len(obs)<32:return scalar_lower(x,obs,par,u,dt,delta,segments)
 if _pool is None:_pool=ThreadPoolExecutor(max_workers=min(4,os.cpu_count() or 1),thread_name_prefix='cbf_certificate')
 # No rows are dropped; collect in original obstacle order. Numba releases GIL.
 cuts=np.linspace(0,len(obs),min(4,os.cpu_count() or 1)+1,dtype=int)
 jobs=[_pool.submit(scalar_lower,x,obs[a:b],par,u,dt,delta,segments) for a,b in zip(cuts[:-1],cuts[1:])]
 return np.concatenate([q.result() for q in jobs])

def candidate_lower(x,obs,par,u,dt,delta,segments):
 from runtime_optimizer import STATS
 start=time.perf_counter()
 try:return _compute(x,obs,par,u,dt,delta,segments)
 finally:STATS['candidate_certificate_ms']=STATS.get('candidate_certificate_ms',0.)+(time.perf_counter()-start)*1000
