"""Freeze actual reference trajectories and their provenance; no controller import."""
from pathlib import Path
import argparse,sys,json,hashlib,csv
import numpy as np
ROOT=Path(__file__).resolve().parent
DEFAULT_REFERENCE=Path(r'F:\CBFbizhang\safe_control-main\3D_RA_DPCBF+HOCBF')

def save(name,tape,dt,start,goal,workspace,waypoints,source,details):
    dest=ROOT/'reference_inputs';dest.mkdir(exist_ok=True)
    path=dest/(name+'.npz')
    np.savez_compressed(path,states=tape,dt=dt,start=start,goal=goal,workspace=workspace,waypoints=waypoints)
    dp=tape[1:,:,:3]-tape[:-1,:,:3]
    velocity_defect=dp/dt-.5*(tape[1:,:,3:6]+tape[:-1,:,3:6])
    row=dict(name=name,source=str(source),details=details,obstacles=tape.shape[1],frames=len(tape),dt=dt,
        duration_s=(len(tape)-1)*dt,workspace=workspace.tolist(),start=start.tolist(),goal=goal.tolist(),
        max_logged_acceleration=float(np.linalg.norm(tape[:,:,6:9],axis=2).max()),
        max_velocity_increment_per_s=float(np.linalg.norm(np.diff(tape[:,:,3:6],axis=0),axis=2).max()/dt),
        max_discrete_kinematic_defect_m_s=float(np.linalg.norm(velocity_defect,axis=2).max()),
        source_states_sha256=hashlib.sha256(tape.tobytes()).hexdigest(),file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        reference_knots_retained_exactly=True,
        interpolation='quintic Hermite through original p,v,a; instantaneous reference reflections are not bounded-acceleration evidence')
    (dest/(name+'.json')).write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
    print(name,tape.shape,flush=True)
    return row

def main(reference):
    sys.path.insert(0,str(reference));sys.path.insert(0,str(reference/'results/6'));sys.path.insert(0,str(reference/'results/7'))
    from ra_dpcbf_3d.scenarios import build_main_scenario,build_showcase_scenario
    from breakout7.scenario import build_surrounded10_tape
    rows=[]
    workspace=np.array([[0.,22.],[0.,15.],[0.,6.]])
    start=np.array([1.,7.5,3.,1.,0.,0.]);goal=np.array([20.,7.5,3.])
    for name,folder in [('default8','default8'),('surrounded10','surrounded10'),('dense50','n50_r07_seed10000')]:
        source=reference/'results/6/preflight_3d'/folder/'obstacle_tapes.npz'
        with np.load(source) as a:tape=a['lifted_3d_tape']
        rows.append(save(name,tape,.05,start,goal,workspace,goal[None,:],source,'Original frozen 3D reference tape, every saved p/v/a/r sample unchanged.'))
    bundle=build_surrounded10_tape(duration=35.)
    rows.append(save('breakout10',bundle.states,bundle.dt,bundle.start_state,bundle.goal,bundle.workspace,bundle.goal[None,:],reference/'results/7/breakout7/scenario.py','Original seed=17 breakout generator; full horizon, not terminal-only log.'))
    factories=[('main10',lambda:build_main_scenario(7))]
    for profile in range(1,6):
        seed=20266214 if profile==5 else 7
        factories.append((f'profile{profile}',lambda profile=profile,seed=seed:build_showcase_scenario(profile,seed)))
    for name,factory in factories:
        scenario=factory();dt=.08;steps=int(40/dt)
        tape=np.empty((steps+1,len(scenario.obstacles),10))
        for k in range(steps+1):
            tape[k]=scenario.obstacle_array()
            if k<steps:
                for obstacle in scenario.obstacles:obstacle.step(dt)
        rows.append(save(name,tape,dt,scenario.start_state,scenario.goal,scenario.workspace,scenario.waypoints,
            reference/'ra_dpcbf_3d/scenarios.py',f'Original native scene factory; profile={scenario.profile}, seed={scenario.seed}; dt=.08.'))
    (ROOT/'reference_inputs/manifest.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,default=DEFAULT_REFERENCE);a=p.parse_args();main(a.reference)
