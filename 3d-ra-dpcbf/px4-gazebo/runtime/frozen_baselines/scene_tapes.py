"""Exact saved reference knots or continuous bounded-motion counterparts."""
from pathlib import Path
import json,os,tempfile
os.environ.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'ra_reference_20261016'))
import numpy as np
from numba import njit
from dynamic_scenarios import build_tape,manoeuvre_schedule,TAPE_DT
ROOT=Path(__file__).resolve().parent
SCENES=['default8','surrounded10','dense50','breakout10','main10']+[f'profile{i}' for i in range(1,6)]+['corridor_crossing']

@njit(cache=True)
def state_at(tape,t,tape_dt,mode):
    if t>=(len(tape)-1)*tape_dt-1e-12:return tape[-1].copy()
    k=min(int(np.floor((t+1e-10)/tape_dt)),len(tape)-2)
    tau=max(0.,min(1.,(t-k*tape_dt)/tape_dt));h=tape_dt;a=tape[k];b=tape[k+1];out=a.copy()
    if mode==2:
        # Literal discrete reference: original states are applied at their native
        # update instants. Between updates use the reported instantaneous p,v,a.
        # The original reflection/contact resets remain discontinuities; do not
        # invent enormous C2 acceleration pulses to hide those resets.
        dt=tau*h
        out[:,:3]=a[:,:3]+dt*a[:,3:6]+.5*dt*dt*a[:,6:9]
        out[:,3:6]=a[:,3:6]+dt*a[:,6:9]
    elif mode==1:
        dt=tau*h;jerk=(b[:,6:9]-a[:,6:9])/h
        out[:,:3]=a[:,:3]+dt*a[:,3:6]+.5*dt**2*a[:,6:9]+dt**3/6*jerk
        out[:,3:6]=a[:,3:6]+dt*a[:,6:9]+.5*dt**2*jerk
        out[:,6:9]=a[:,6:9]+dt*jerk
    else:
        # C2 Hermite interpolation, passing every original p/v/a sample.
        # Large acceleration around original impact resets is explicitly audited.
        c0=a[:,:3];c1=h*a[:,3:6];c2=.5*h*h*a[:,6:9]
        dp=b[:,:3]-c0-c1-c2;dv=h*b[:,3:6]-c1-2*c2;da=h*h*b[:,6:9]-2*c2
        c3=10*dp-4*dv+.5*da;c4=-15*dp+7*dv-da;c5=6*dp-3*dv+.5*da
        out[:,:3]=c0+tau*c1+tau**2*c2+tau**3*c3+tau**4*c4+tau**5*c5
        out[:,3:6]=(c1+2*tau*c2+3*tau**2*c3+4*tau**3*c4+5*tau**4*c5)/h
        out[:,6:9]=(2*c2+6*tau*c3+12*tau**2*c4+20*tau**3*c5)/(h*h)
    return out

def load_scene(name,mode='bounded',seed=17,duration=35.):
    if name=='reselection3d':
        from reselection_scene import load_reselection
        return load_reselection(mode)
    if name=='corridor_crossing':
        from corridor_scene import load_corridor
        return load_corridor(mode)
    if name in ('head_on_static','head_on_dynamic','diagonal','diagonal_ceiling'):
        direction=np.array([1.,0.,0.]) if name.startswith('head_on') else np.array([1.,1.,0.])/np.sqrt(2)
        start=np.r_[np.array([2.,4.,3.]),direction];goal=start[:3]+14*direction
        obstacle=start[:3]+6*direction;velocity=-.45*direction if name=='head_on_dynamic' else np.zeros(3)
        radii=np.array([.5]);p=obstacle[None,:];v=velocity[None,:]
        if name=='diagonal_ceiling':
            p=np.r_[p,(obstacle+np.array([0,0,1.9]))[None,:]];v=np.r_[v,np.zeros((1,3))];radii=np.r_[radii,.65]
        workspace=np.array([[0.,22.],[0.,20.],[0.,7.]])
        steps=int(np.ceil(duration/.01))+3;tape=np.zeros((steps,len(p),10))
        t=np.arange(steps)*.01;tape[:,:,:3]=p+t[:,None,None]*v;tape[:,:,3:6]=v;tape[:,:,9]=radii
        return dict(tape=tape,tape_dt=.01,mode=1,workspace=workspace,start=start,goal=goal,waypoints=goal[None,:],
                    source='exact symmetric explanatory scene',same_reference_knots=False)
    path=ROOT/'reference_inputs'/f'{name}.npz';data=dict(np.load(path));original=data['states'];td=float(data['dt'])
    if mode=='reference':tape=original;kind=2
    elif mode=='reference_c2':tape=original;kind=0
    else:
        commands,counts=manoeuvre_schedule(original.shape[1],duration+.02,seed)
        tape=build_tape(original[0,:,:3],original[0,:,3:6],original[0,:,9],data['workspace'],commands)
        td=TAPE_DT;kind=1
    return dict(tape=tape,tape_dt=td,mode=kind,workspace=data['workspace'],start=data['start'],goal=data['goal'],
                waypoints=data['waypoints'],source=str(path.relative_to(ROOT)),same_reference_knots=mode in ('reference','reference_c2'))

def make_config():
    path=ROOT/'method_config.json';cfg=json.loads(path.read_text(encoding='utf8'))
    p=cfg['parameters'][:20]+[0.,.3,.15]
    cfg.update(parameters=p,prediction_margin=0.,robot_radius=.3,base_margin=.15,
      detour={'k_perp':1.2,'nu_req':.3,'eta':.6,'s_on':4.,'gamma_min':.04,'tau_off':.4,
              'direction_method':'exact feasible-polytope vertices; held direction support by LP',
              'sampled_detour_margin':True},mechanism_version='user_equations_20261016')
    path.write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf8');return cfg
