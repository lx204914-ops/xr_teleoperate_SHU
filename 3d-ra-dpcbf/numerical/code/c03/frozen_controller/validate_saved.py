"""Read back all arrays; independently check the executable detour and provenance."""
from pathlib import Path
import json,hashlib,argparse
import numpy as np
from detour import polytope,support,vertices
from scene_tapes import state_at
from configuration import output_root
ROOT=Path(__file__).resolve().parent

def check(folder):
    file=folder/'history.npz';a=dict(np.load(file));m=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
    h=a['history'];b=a['barriers'];o=a['obstacles'];d=a['detour'];p=a['parameters'];active=d[:,0]>0
    tape_hash=hashlib.sha256(a['obstacle_tape'].tobytes()).hexdigest()
    assert tape_hash==m['obstacle_tape_sha256']
    assert np.isfinite(h[:,:7]).all() and np.isfinite(b[:,:,:12]).all()
    r=o[:,:,:3]-h[:,None,1:4];w=o[:,:,3:6]-h[:,None,4:7];D=np.linalg.norm(r,axis=2);e=r/D[:,:,None]
    q=np.sum(w*e,axis=2);z=w-q[:,:,None]*e;Z=np.sum(z*z,axis=2);s=D-b[:,:,3];beta=p[1]/(p[0]*p[19])
    independent=q-b[:,:,4]+s/p[0]+beta*s*Z
    assert abs(independent-b[:,:,7]).max()<1e-10
    B=-e-2*beta*s[:,:,None]*z
    assert abs(B-b[:,:,9:12]).max()<1e-10
    assert np.linalg.norm(B,axis=2).min()>=1-1e-10
    assert abs(D-b[:,:,0]).max()<1e-10
    gamma_error=0.;chi_error=0.;norm_error=0.;orth_error=0.;bound_slack=1e20;gain=m['config']['detour']['k_perp']
    for k in np.flatnonzero(active)[::max(1,int(active.sum()/24))]:
        j=int(d[k,1]);direction=d[k,2:5];chi=-direction@z[k,j]
        chi_error=max(chi_error,abs(chi-d[k,5]));norm_error=max(norm_error,abs(np.linalg.norm(direction)-1))
        orth_error=max(orth_error,abs(direction@e[k,j]))
        A,c=polytope(B[k],b[k,:,8]-a['sampling_margins'][k]);value,u=support(A,c,direction)
        expected=value-direction@o[k,j,6:9]+(gain-q[k,j]/D[k,j])*chi
        gamma_error=max(gamma_error,abs(expected-d[k,6]))
        assert d[k,8]<=d[k,7]+1e-7
    for segment in np.unique(d[active,10]):
        ids=np.flatnonzero(active&(d[:,10]==segment));k=ids[0];elapsed=h[ids,0]-h[k,0]
        nu=d[ids,8].min();bound=np.exp(-gain*elapsed)*d[k,5]+nu/gain*(1-np.exp(-gain*elapsed))
        bound_slack=min(bound_slack,float((d[ids,5]-bound).min()))
    assert gamma_error<1e-7 and chi_error<1e-10 and norm_error<1e-9 and orth_error<1e-9
    # Fine-step transport discretizes the continuous transport equation.
    if int(a['motion_mode'])==1:assert bound_slack>=-2e-5
    base=p[21]+p[22]+o[:,:,9]
    actual_generator=bool(np.min(b[:,:,3]-base)>=-1e-9 and np.max(b[:,:,3]-base)<=p[5]+1e-9 and
                          np.max(abs(b[:,:,4]))<=p[2]*p[5]+1e-9)
    assert actual_generator
    scope_ok=int(a['motion_mode'])==1 and m['sampled_data_certificate_applied'] and m['min_dynamic_clearance_m']>=-1e-7 and m['min_h']>=-1e-7 and m['min_intersample_CBF_residual']>=-1e-7
    row=dict(scene=m['scene'],mode=m['mode'],variant=m['variant'],termination=m['termination'],
        source_npz_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),loaded_all_arrays=True,
        formula_h_max_error=float(abs(independent-b[:,:,7]).max()),control_coefficient_max_error=float(abs(B-b[:,:,9:12]).max()),
        Gamma_support_LP_max_error=gamma_error,chi_max_error=chi_error,direction_norm_max_error=norm_error,
        direction_orthogonality_max_error=orth_error,conditional_chi_bound_min_slack=bound_slack if bound_slack<1e19 else None,
        conditional_chi_bound_continuous_motion_hypothesis=int(a['motion_mode'])==1,
        radius_interval_and_rate_checks=True,dynamic_distance_claim_supported_by_this_run=scope_ok,
        Zmax_domain_respected=bool(m['max_interval_Z_bound']<=p[19]),
        sufficient_feasibility_theorems_not_assumed_to_hold_globally=True)
    (folder/'validation.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8');return row

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('path',nargs='?',default='results/final');args=parser.parse_args()
    dest=output_root(args.path)
    folders=list(dest.rglob('history.npz'))
    if not folders:raise SystemExit('No saved histories found; no validation has been claimed.')
    rows=[check(f.parent) for f in folders]
    (dest/'validation_summary.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({'validated_runs':len(rows),'bounded_distance_checks':sum(r['dynamic_distance_claim_supported_by_this_run'] for r in rows)},ensure_ascii=False))
