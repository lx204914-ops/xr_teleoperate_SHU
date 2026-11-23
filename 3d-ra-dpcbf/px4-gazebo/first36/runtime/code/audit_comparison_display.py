from pathlib import Path
import sys,json
import numpy as np
from native_audit_reader import records
folder=Path(sys.argv[1]);summary=json.loads((folder/'summary.json').read_text());view=json.loads((folder/'native_view_status.json').read_text());method=summary['method']
assert view['method']==method and view['risk_halos_enabled']==(method=='ra')
rows,_=records(folder/'native_frames.bin');error=0.;samples=0
if (folder/'display_snapshots.npz').exists():
 with np.load(folder/'display_snapshots.npz',allow_pickle=False) as z:
  states=z['state'];obs=z['obstacles'];bars=z['barriers'];params=z['parameters']
  sys.path.insert(0,str(folder.parents[1]/'frozen_controller'));sys.path.insert(0,str(folder.parents[1]/'frozen_baselines'))
  from baseline_coefficients import KL,KM
  par=np.asarray(json.loads((folder.parents[1]/'frozen_controller/method_config.json').read_text())['parameters'])
  for x,o,b,p in zip(states,obs,bars,params):
   if method=='ra':mu=b[:,6]/par[0]-b[:,4];lam=par[1]*b[:,6]/(par[0]*par[19])
   elif method=='park':
    D=np.linalg.norm(o[:,:3]-x[:3],axis=1);d=np.sqrt(np.maximum(D*D-(par[21]+par[22]+o[:,9])**2,1e-12));W=np.sqrt(np.sum((o[:,3:6]-x[3:])**2,axis=1)+1e-12);mu=KM*d;lam=KL*d/W
   else:mu=np.zeros(len(o));lam=np.zeros(len(o))
   error=max(error,float(np.max(abs(mu-p[:,0]))),float(np.max(abs(lam-p[:,1]))));samples+=1
assert error<1e-8
result=dict(method=method,passed=bool(samples and len(rows)>1),snapshots=samples,native_frames=len(rows),maximum_parameter_error=error,risk_halos_only_RA=True,geometry=view['geometry_kind'],scope='C3BF fixed-size cones are labelled direction schematics; no exact feasible-set claim. MA has no RA surface; Park coefficients match actual baseline formula.')
(folder/'relative_display_check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
