"""Offline checks for an explicitly supplied current/initial state; no rollout."""
import argparse,json
from pathlib import Path
import numpy as np
from configuration import load_config,output_root,config_sha256
from dynamic_study.kernel import constraints,radius_ref
from feasibility_checks import check_initial_state,check_current_sets
from sampled_safety import interval_margins


def check(data,config,horizon=None):
    x=np.asarray(data['robot'],dtype=float);obs=np.asarray(data['obstacles'],dtype=float)
    if x.shape!=(6,) or obs.ndim!=2 or obs.shape[1]!=10 or not len(obs):
        raise ValueError('Provide robot[6] and nonempty obstacles[N,10].')
    if not np.isfinite(x).all() or not np.isfinite(obs).all():raise ValueError('Nonfinite state.')
    p=np.array(config['parameters'])
    if 'boundary_states' in data:
        states=data['boundary_states']
        if len(states)!=len(obs):raise ValueError('One radius state per obstacle is required.')
        R=np.array([v['radius'] for v in states],dtype=float)
        V=np.array([v['rate'] for v in states],dtype=float)
        ref=np.array([v['reference'] for v in states],dtype=float)
    elif data.get('initialization')=='explicit_R0_zero_rate':
        R=p[21]+p[22]+obs[:,9];V=np.zeros(len(obs));ref=np.empty(len(obs))
        for i in range(len(obs)):
            r=obs[i,:3]-x[:3];D=np.linalg.norm(r)
            if D<=0:raise ValueError('LOS undefined at zero center distance.')
            q=r@(obs[i,3:6]-x[3:])/D
            ref[i]=radius_ref(q,R[i],p)[0]
    else:
        raise ValueError('Supply actual boundary_states, or explicitly request initial R0/zero rate; no state is reset implicitly.')
    B,C,log=constraints(x,obs,R,V,ref,p,5)
    margin,zupper,L=interval_margins(log,R,V,ref,p,config['control_dt'],5)
    initial=check_initial_state(x,obs,R,V,ref,config,horizon)
    sets=check_current_sets(B,C,config,margin)
    observed=bool(np.max(np.linalg.norm(obs[:,6:9],axis=1))<=config['obstacle_acceleration_bound']+1e-10)
    return dict(check_completed=True,method_config_sha256=config_sha256(config),
        initial_conditions_and_sufficient_certificates=initial,current_sets=sets,
        observed_acceleration_within_declared_bound=observed,
        interval_Z_upper=zupper.tolist(),interval_L=L.tolist(),sampling_margins=margin.tolist(),
        reference_matches_current_law=all(r['reference_matches_current_law'] for r in initial['initialization']['rows']),
        control_computed=False,trajectory_generated=False,
        scope='This is an explicit-state mathematical check, not a closed-loop navigation or Monte Carlo experiment.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--state',type=Path,required=True)
    p.add_argument('--horizon',type=float);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();data=json.loads(a.state.read_text(encoding='utf-8-sig'))
    result=check(data,load_config(),a.horizon)
    from run_suite import json_safe
    result=json_safe(result);dest=output_root(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
    dest.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
    print(json.dumps(dict(report=str(dest),check_completed=True,
        initial_common_margin=result['initial_conditions_and_sufficient_certificates']['initial_common_margin']['status'],
        finite_interval=result['initial_conditions_and_sufficient_certificates']['finite_interval']['status']),ensure_ascii=False))

if __name__=='__main__':main()
