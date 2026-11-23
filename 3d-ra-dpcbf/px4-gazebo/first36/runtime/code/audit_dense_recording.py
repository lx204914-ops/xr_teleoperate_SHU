from pathlib import Path
import json,hashlib
import numpy as np
from PIL import Image,ImageSequence
from native_audit_reader import records

def audit(root,name):
    root=Path(root);folder=root/'native_recording'/name;run=root/'runs'/name
    meta=json.loads((folder/'recording_receipt.json').read_text())
    raw,_=records(run/'native_frames.bin');assert raw
    frames=np.asarray(raw);ids=frames[:,2].astype(int)
    assert np.array_equal(ids,np.arange(1,len(ids)+1))
    lookup={int(v[2]):v for v in frames}
    with np.load(run/'history.npz',allow_pickle=False) as z:bars=z['barriers'];h=z['history'];obs=z['obstacles']
    max_h=0.;max_source=0.;seen=[]
    for record in meta['records']:
        before=lookup[record['native_frame_before']];after=lookup[record['native_frame_after']]
        assert before[11]==after[11]==0
        assert abs(before[1]-record['sim_before_s'])<1e-10 and abs(after[1]-record['sim_after_s'])<1e-10
        assert int(after[2])>=int(before[2])
        n=int(before[6]);assert n in (50,100)
        params=before[12:].reshape(n,33)
        assert np.max(abs(params[:,5]-np.asarray(record['curvature_vertex'])[:,0]))<1e-12
        assert np.max(abs(params[:,4]-np.asarray(record['curvature_vertex'])[:,1]))<1e-12
        assert np.max(abs(params[:,6]-record['risk_radius']))<1e-12
        for row in (before,after):
            p=row[12:].reshape(n,33);k=round(row[1]/.02);b=bars[k]
            mu=b[:,6]/p[:,15]-b[:,4];lam=p[:,16]*b[:,6]
            max_source=max(max_source,float(np.max(abs(mu-p[:,4]))),float(np.max(abs(lam-p[:,5]))),float(np.max(abs(b[:,3]-p[:,6]))))
            e=p[:,1:4];probe=p[:,18:27].reshape(n,3,3)/p[:,14,None,None]
            q=np.sum(probe*e[:,None,:],axis=2);z=probe-q[:,:,None]*e[:,None,:]
            max_h=max(max_h,float(np.max(abs(q+p[:,4,None]+p[:,5,None]*np.sum(z*z,axis=2)))))
            seen.append(int(row[2]))
    assert max_h<1e-9 and max_source<1e-9
    gif=folder/'gazebo_native_live.gif';unique=set();duration=0
    with Image.open(gif) as im:
        count=im.n_frames
        for f in ImageSequence.Iterator(im):
            unique.add(hashlib.sha256(f.convert('RGB').tobytes()).hexdigest());duration+=f.info.get('duration',0)
    assert count==meta['frames']==len(meta['records']) and len(unique)>1
    assert abs(duration/1000-meta['playback_duration_s'])<.011
    assert hashlib.sha256(gif.read_bytes()).hexdigest()==meta['sha256']
    result=dict(passed=True,run=name,GIF_frames=count,unique_GIF_frames=len(unique),duration_s=duration/1000,
        native_ledger_frames=len(ids),missing_native_ledger_frames=0,GIF_frames_with_before_after_audits=count,
        maximum_GIF_mesh_h_residual=max_h,maximum_GIF_controller_parameter_error=max_source,
        sha256=meta['sha256'],scope='Exact native C++ render ledger brackets every XWD screenshot. Includes final capture bracket after display close, without rewriting the original pre-close NPZ. No pixel-capture atomicity claim.')
    (folder/'frame_coverage_check.json').write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    root=Path(__file__).resolve().parents[1];selection=json.loads((root/'selected_runs.json').read_text())
    print(json.dumps([audit(root,v) for v in selection.values()],indent=2))
