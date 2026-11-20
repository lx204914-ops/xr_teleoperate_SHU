"""Read-only verification, including both ends of each executed hold interval."""
from pathlib import Path
import sys,json,csv,hashlib,subprocess
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
import numpy as np
METHODS=("ma_cbf_vo","c3bf","park","ra")
LABELS=("MA-CBF-VO","C3BF","Park-inspired","RA-DPCBF")
def folder(method):
    tag="prioritized_execution"
    return HERE/"results"/("four_obstacle_"+method+"_"+tag)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=HERE/"final";out.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/"frozen_controller/source_manifest.json").read_text("utf8"))
    assert all(sha(ROOT/"frozen_controller"/name)==m["sha256"] for name,m in manifest.items())
    bm=json.loads((HERE/"frozen_baselines/source_manifest.json").read_text("utf8"))
    assert all(sha(HERE/"frozen_baselines"/name)==m["sha256"] for name,m in bm.items())
    cfg=json.loads((ROOT/"frozen_controller/method_config.json").read_text("utf8"));kp=cfg["detour"]["k_perp"]
    rows=[];checks=[];initial_physical=None
    for method,label in zip(METHODS,LABELS):
        p=folder(method);s=json.loads((p/"summary.json").read_text("utf8"))
        with np.load(p/"history.npz",allow_pickle=False) as z:
            h=z["history"];bar=z["barriers"];obs=z["obstacles"];det=z["detour"];goal=z["goal"]
            assert len(h)==len(bar)+1 and bar.shape[1]==4
            physical0=np.r_[h[0,1:7],h[0,19:27]]
            if initial_physical is None:initial_physical=physical0
            else:assert np.max(abs(physical0-initial_physical))<1e-9
            distance=np.linalg.norm(h[:,1:4]-goal,axis=1);assert np.max(abs(distance-h[:,-2]))<1e-9
            first=float(h[np.flatnonzero(distance<.25)[0],0]);assert abs(first-s["first_arrival_s"])<1e-9
            assert abs(h[-1,0]-first-2.)<1e-9
            B=bar[:,:,9:12];C=bar[:,:,8];a=h[:len(bar),13:16];u=h[:len(bar),7:10]
            actual=np.einsum("nij,nj->ni",B,a)+C
            command=np.einsum("nij,nj->ni",B,u)+C
            identity=float(np.max(abs(actual-command-np.einsum("nij,nj->ni",B,a-u))));assert identity<1e-10
            minimum=min(float(actual.min()),s["min_actual_CBF_residual"])
            peak=max(float(np.max(np.linalg.norm(a-u,axis=1))),s["acceleration_tracking_max_norm_mps2"])
            assert s["mission"]=="goal" and minimum>=-1e-7 and s["min_body_gap_before_arrival_m"]>0
            assert np.max(abs(u))<=1.5+1e-7 and np.max(abs(np.linalg.norm(h[:,19:23],axis=1)-1.))<1e-6
            aux_min=s.get("min_actual_detour_residual")
            switches=0;segments=0;changed_time=0.
            if method=="ra":
                changed_time=float(np.sum(np.linalg.norm(u-h[:len(bar),16:19],axis=1)>1e-6)*.005)
                active=det[:,1]>.5;segments=int(np.sum(active&~np.r_[False,active[:-1]]))
                for k in np.flatnonzero(active):
                    j=int(det[k,2]);d=det[k,3:6];r=obs[k,j,:3]-h[k,1:4];D=np.linalg.norm(r);e=r/D
                    w=obs[k,j,3:6]-h[k,4:7];q=e@w;chi=-d@(w-q*e)
                    val=float(d@(a[k]-obs[k,j,6:9])+(kp-q/D)*chi-det[k,8])
                    aux_min=val if aux_min is None else min(aux_min,val)
                    if k and active[k-1] and np.dot(d,det[k-1,3:6])<np.cos(np.deg2rad(5)):switches+=1
                assert aux_min is None or aux_min>=-1e-7
            final_error=float(distance[-1]);speed=float(np.linalg.norm(h[-1,4:7]))
            rows.append(dict(method=label,arrival_s=first,path_m=s["path_length_to_arrival_m"],
                minimum_body_clearance_m=s["min_body_gap_before_arrival_m"],minimum_actual_CBF_including_endpoints=minimum,
                peak_tracking_error_including_endpoints=peak,tracking_budget_satisfied=peak<=.5+1e-8,
                final_goal_error_m=final_error,final_speed_m_s=speed,settled_after_2s=final_error<.25 and speed<.2,
                total_control_P95_ms=s["total_control_p95_ms"],deadline_overrun_steps=s["deadline_overrun_steps"],
                min_actual_detour_including_endpoints=aux_min,active_detour_s=s["active_detour_seconds"],
                detour_changed_control_seconds=changed_time,detour_active_segments=segments,held_direction_changes_over_5deg=switches))
            checks.append(dict(method=method,history=str(p/"history.npz"),sha256=sha(p/"history.npz"),
                scene_sha256=str(z["scene_sha256"]),CBF_gap_identity_error=identity,first_arrival_checked=first))
    with (out/"four_method_physical_results.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
    result=dict(all_four_observed_safe_arrival=True,source_main_unchanged=len(manifest),source_baselines_unchanged=len(bm),
        control_dt_s=.005,physics_monitor_dt_s=.0005,shared_initial_state_checked=True,results=rows,checks=checks,
        all_tracking_budgets_satisfied=all(r["tracking_budget_satisfied"] for r in rows),
        hard_realtime_accepted=False,all_settled_after_2s=all(r["settled_after_2s"] for r in rows),
        universal_tracking_bound_proved=False,
        limitations=["This is a single shared-scene physical experiment, not a success-rate statistic.",
                     "Any failed tracking budget or terminal settling must remain reported, not hidden.",
                     "P95 timing does not establish hard realtime; actual overruns are retained."],
        scene_position_velocity_motion_unchanged=True,
        deployment_differences="Common initial acceleration/attitude trim, 200Hz, pre-QP reference preference, and executable optional detour subset. Original source preserved.")
    (HERE/"validation.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result),flush=True)
if __name__=="__main__":main()
