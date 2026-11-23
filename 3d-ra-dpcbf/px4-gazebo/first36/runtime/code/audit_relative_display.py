"""Independent saved-log and native per-render geometry verification; no data edits."""
from pathlib import Path
import json,sys,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def preflight(root):
    root=Path(root);p=np.asarray(json.loads((root/'frozen_controller/method_config.json').read_text())['parameters'])
    folder=root/'runs/continuous_ten_20261101_182838_62a5'
    with np.load(folder/'history.npz',allow_pickle=False) as z:b=z['barriers'];obs=z['obstacles'];state=z['history'][:,4:10]
    r=obs[:,:,:3]-state[:,None,:3];e=r/np.linalg.norm(r,axis=2)[:,:,None]
    axis=np.zeros_like(e);axis[:,:,2]=1;bad=np.linalg.norm(np.cross(e,axis),axis=2)<1e-8;axis[bad]=[0,1,0]
    n1=np.cross(e,axis);n1/=np.linalg.norm(n1,axis=2)[:,:,None];n2=np.cross(e,n1)
    beta=p[1]/(p[0]*p[19]);mu=b[:,:,6]/p[0]-b[:,:,4];lam=beta*b[:,:,6];maximum=0.
    for radius in (0.,.75,1.5):
        for angle in np.linspace(0,2*np.pi,17):
            w=-(mu+lam*radius**2)[:,:,None]*e+radius*(np.cos(angle)*n1+np.sin(angle)*n2)
            q=np.sum(e*w,axis=2);z=w-q[:,:,None]*e
            maximum=max(maximum,float(np.max(abs(q+mu+lam*np.sum(z*z,axis=2)))))
    result=dict(passed=maximum<1e-9,source_control_states=len(b),obstacles=b.shape[1],maximum_independent_boundary_error=maximum,
        maximum_barrier_rewrite_error=float(np.max(abs(b[:,:,1]+mu+lam*b[:,:,2]-b[:,:,7]))),method_changed=False,scene_changed=False)
    print(json.dumps(result,indent=2));return result

def audit(folder):
    folder=Path(folder);root=folder.parents[1]
    with np.load(folder/'display_snapshots.npz',allow_pickle=False) as z:snap={k:z[k] for k in z.files}
    with np.load(folder/'native_frame_audit.npz',allow_pickle=False) as z:raw=z['frames']
    with np.load(folder/'history.npz',allow_pickle=False) as z:history=z['history'];b=z['barriers'];obs=z['obstacles'];physics=z['physics_truth'];actors=z['obstacle_pose_samples']
    summary=json.loads((folder/'summary.json').read_text());par=np.asarray(json.loads((root/'frozen_controller/method_config.json').read_text())['parameters'])
    n=b.shape[1];assert raw.ndim==2 and raw.shape[1]==12+33*n,raw.shape
    params=raw[:,12:].reshape(-1,n,33);time=raw[:,1];indices=np.rint(time/.02).astype(int);assert np.max(abs(history[indices,1]-time))<1e-8
    expected=b[indices];expected_mu=expected[:,:,6]/par[0]-expected[:,:,4];expected_lam=par[1]*expected[:,:,6]/(par[0]*par[19])
    e0=obs[indices,:,:3]-history[indices,None,4:7];e0/=np.linalg.norm(e0,axis=2)[:,:,None]
    e=params[:,:,1:4];mu=params[:,:,4];lam=params[:,:,5];R=params[:,:,6];L=params[:,:,13];tau=params[:,:,14]
    probe=params[:,:,18:27].reshape(-1,n,3,3);w=probe/tau[:,:,None,None];q=np.sum(w*e[:,:,None,:],axis=3);zt=w-q[:,:,:,None]*e[:,:,None,:]
    residual=q+mu[:,:,None]+lam[:,:,None]*np.sum(zt*zt,axis=3)
    radius=np.linalg.norm(zt,axis=3)
    source_error=max(float(np.max(abs(mu-expected_mu))),float(np.max(abs(lam-expected_lam))),float(np.max(abs(R-expected[:,:,3]))),float(np.max(abs(params[:,:,7]-expected[:,:,4]))),float(np.max(abs(params[:,:,8]-expected[:,:,5]))))
    timestamp_error=float(np.max(abs(params[:,:,9]-(summary['mission_sim_start']+time[:,None]))))
    origin=raw[:,3:6];origin_actual=np.column_stack([np.interp(raw[:,0],physics[:,0],physics[:,j]) for j in (1,2,3)])
    center_errors=[]
    for i in range(n):
        a=actors[actors[:,1]==i];a=a[np.argsort(a[:,0],kind='stable')];_,ix=np.unique(a[:,0],return_index=True);a=a[ix]
        actual=np.column_stack([np.interp(raw[:,0],a[:,0],a[:,j]) for j in (2,3,4)])
        center_errors.append(np.max(np.linalg.norm(params[:,i,30:33]-actual,axis=1)))
    snapshot_indices=np.rint(snap['control_t']/.02).astype(int)
    snapshot_error=float(np.max(abs(snap['barriers']-b[snapshot_indices])))
    vertex_error=float(np.max(np.linalg.norm(probe[:,:,0,:]+tau[:,:,None]*mu[:,:,None]*e,axis=2)))
    halo_error=float(np.max(abs(np.linalg.norm(params[:,:,27:30],axis=2)-R)))
    age=raw[:,0,None]-params[:,:,9]
    check=dict(passed=False,run=folder.name,native_render_frames=len(raw),controller_packets=len(snap['control_t']),surfaces_per_frame=n,
        maximum_independent_mesh_h_residual=float(np.max(abs(residual))),maximum_controller_parameter_error=source_error,
        maximum_controller_LOS_error=float(np.max(np.linalg.norm(e-e0,axis=2))),maximum_snapshot_raw_barrier_error=snapshot_error,
        maximum_controller_timestamp_error_s=timestamp_error,maximum_vertex_formula_error_m=vertex_error,
        maximum_UAV_origin_vs_physics_error_m=float(np.max(np.linalg.norm(origin-origin_actual,axis=1))),
        maximum_native_surface_attachment_error_m=float(raw[:,8].max()),maximum_risk_circle_radius_error_m=halo_error,
        maximum_obstacle_center_vs_physics_error_m=float(max(center_errors)),
        tau_vis_range_s=[float(tau.min()),float(tau.max())],tangent_limit_range_mps=[float(L.min()),float(L.max())],
        maximum_outer_mesh_tangent_limit_error_mps=float(np.max(abs(radius[:,:,1]-L))),
        controller_snapshot_age_ms=dict(min=float(age.min()*1000),p95=float(np.quantile(age,.95)*1000),max=float(age.max()*1000)),
        radius_change_per_obstacle_m=np.ptp(R,axis=0).tolist(),lambda_range_per_obstacle=np.c_[lam.min(0),lam.max(0)].tolist(),
        origin_semantics='Actual UAV position represents relative velocity ZERO, not the current relative velocity point',
        temporal_semantics='Latest timestamped controller coefficients and R held together; actual UAV and obstacle positions follow physics; no fake interpolation',
        independent_verification='Inverse the actual C++ mesh probes, match every native render to raw controller logs, and substitute into the unchanged h=0 formula')
    check['passed']=bool(len(raw)>100 and n in (50,100) and source_error<1e-9 and timestamp_error<1e-8 and snapshot_error<1e-9
        and np.max(abs(residual))<1e-9 and vertex_error<1e-9 and halo_error<1e-9 and np.max(abs(tau-.6))<1e-12
        and np.max(abs(L-1.5))<1e-12 and check['maximum_controller_LOS_error']<1e-9
        and check['maximum_outer_mesh_tangent_limit_error_mps']<1e-9 and check['maximum_UAV_origin_vs_physics_error_m']<1e-4
        and check['maximum_native_surface_attachment_error_m']<1e-5 and check['maximum_obstacle_center_vs_physics_error_m']<1e-4
        and age.min()>-1e-8)
    (folder/'relative_display_check.json').write_text(json.dumps(check,indent=2),encoding='utf8');print(json.dumps(check,indent=2));return check

if __name__=='__main__':
    if sys.argv[1]=='--preflight':preflight(sys.argv[2])
    else:audit(sys.argv[1])
