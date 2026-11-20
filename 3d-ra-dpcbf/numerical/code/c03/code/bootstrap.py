from pathlib import Path
import os,sys,json,hashlib,importlib.util
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
os.environ['NUMBA_CACHE_DIR']=str(ROOT.parent/'cache')
OLD=ROOT/'frozen_original/current_method_conditional'
NEW=ROOT/'frozen_improved/current_method_conditional'
for p in (ROOT/'frozen_original/code',NEW):sys.path.insert(0,str(p))
sys.path.insert(0,str(ROOT/'code'))
# Initial-gate helper imports this module as bootstrap. Preserve its original
# relative HERE/config semantics without reaching the old external drive.
HERE=ROOT/'frozen_original'
def contract():return json.loads((ROOT/'inputs/preregistration.json').read_text('utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,a):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+f'.tmp.{os.getpid()}')
 tmp.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8');tmp.replace(p)
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
METHODS=('original','improved','no_detour','gamma_direction','old_exit','fixed_radius')
LABELS=dict(original='原方法',improved='改进完整方法',no_detour='关闭绕行',gamma_direction='恢复Γ选向',old_exit='恢复原退出',fixed_radius='固定半径')
def verify():
 for name,h in json.loads((ROOT/'inputs/preserved_sources.json').read_text('utf8')).items():assert sha(name)==h,name
 frozen=ROOT/'inputs/run_manifest.json'
 if frozen.exists():
  for name,h in json.loads(frozen.read_text('utf8')).items():assert sha(name)==h,name
