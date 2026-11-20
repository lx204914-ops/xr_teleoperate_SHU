"""Shared deployment-only ground envelope, included BEFORE every safety QP.

Use existing Tr/kh gains. No post-QP clipping or new obstacle safety formula.
The interval lower bound assumes the existing norm tracking budget (0.5),
2.0 m/s^2 vertical actual-acceleration envelope and a 20ms held command.
State-estimation error is NOT certified by this implementation.
"""
import numpy as np

def rows(controller,x):
    B=np.r_[np.eye(3),-np.eye(3)];C=np.full(6,1.5)
    if controller.config.get('shared_execution_slew',False):
        from shared_execution_box import proposed_rows
        C=proposed_rows(controller)
    if not hasattr(controller,'ground_minimum_estimated_z'):
        # Fixed-size warm-up workspace, before world/estimator frame is known.
        return np.r_[B,[[0.,0.,1.]]],np.r_[C,1.e6]
    k1=1./controller.par[0];k2=controller.par[3];dt=controller.config['control_dt']
    height=float(x[2]-controller.ground_minimum_estimated_z);vz=float(x[5])
    hold=(k1+k2)*2.*dt+k1*k2*(abs(vz)*dt+dt*dt)
    coefficient=(k1+k2)*vz+k1*k2*height-.5-hold
    controller.ground_diagnostic=dict(height=height,vertical_velocity=vz,k1=float(k1),k2=float(k2),hold_margin=float(hold),tracking_margin=.5,C=float(coefficient),h1=float(vz+k1*height))
    return np.r_[B,[[0.,0.,1.]]],np.r_[C,coefficient]

def audit(folder):
    from pathlib import Path
    import json
    p=Path(folder);s=json.loads((p/'summary.json').read_text())
    z=np.load(p/'history.npz');phys=z['physics_truth'];h=z['history']
    m=(phys[:,0]>=s['mission_sim_start'])&(phys[:,0]<=s['mission_end_sim']);ph=phys[m]
    if not len(ph):
        result=dict(has_mission_physics=False,passed=None)
    else:
        cfg=json.loads((p.parents[1]/'frozen_controller/method_config.json').read_text());k1=1./cfg['parameters'][0];k2=cfg['parameters'][3]
        height=ph[:,3]-.4;h1=ph[:,6]+k1*height;phi=ph[:,9]+(k1+k2)*ph[:,6]+k1*k2*height
        result=dict(has_mission_physics=True,minimum_world_z=float(ph[:,3].min()),minimum_height_above_envelope=float(height.min()),minimum_h1=float(h1.min()),minimum_actual_ground_CBF=float(phi.min()),sample_scope='saved Gazebo physics samples; state-estimation robustness not certified',passed=bool(height.min()>-1e-6 and h1.min()>-1e-6 and phi.min()>-1e-6))
    for fname in ('independent_check.json','continuous_check.json'):
        f=p/fname
        if f.exists():
            a=json.loads(f.read_text());a['obstacle_and_execution_check_before_ground']=a['passed'];a['ground_envelope_passed']=result['passed'];a['passed']=bool(a['passed'] and result['passed']);f.write_text(json.dumps(a,indent=2))
    (p/'ground_check.json').write_text(json.dumps(result,indent=2));return result
