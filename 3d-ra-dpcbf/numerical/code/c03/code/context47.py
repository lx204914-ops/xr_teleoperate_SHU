from pathlib import Path
import os,sys,json,hashlib,time
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']=str(ROOT.parent/'cache')
sys.path.insert(0,str(ROOT/'code'))
import bootstrap
METHODS=('full','no_detour','no_reselection','no_progress_guard','fixed_radius','no_transverse','fixed_strength')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+f'.tmp.{os.getpid()}')
 q.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');q.replace(p)
def verify():
 for p,h in json.loads((ROOT/'inputs/package_sources.json').read_text('utf8')).items():assert sha(ROOT/p)==h,p
