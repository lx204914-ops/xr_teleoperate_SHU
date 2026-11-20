"""One identical feasible physical initialization, no scene/outcome search."""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.path.insert(0,str(ROOT))
from deployed_controller import DeployedController
from baseline_adapter import BaselineController,hard_hold,hold_certificates
from sparse_projection import SparseProjectionQP
import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.crazyflie_params import quad_params
def common_trim(initial,goal,obs):
    Blist=[];Clist=[]
    c=DeployedController();c.reset(initial,goal,obs);c.execution_acceleration=np.zeros(3)
    _,status,d=c.step(0.,initial,obs)
    if "B" not in d:raise RuntimeError("RA initial domain failure")
    Blist.append(d["B"]);Clist.append(d["C"]-d["sampling_margin"])
    for name in ("ma_cbf_vo","c3bf","park"):
        b=BaselineController(name);b.reset(initial,goal,obs)
        B,C,_=b.raw_coefficients(initial,obs)
        if name=="ma_cbf_vo":m,_=hard_hold(initial,obs,b.par,b.config["control_dt"])
        else:m,_=hold_certificates(initial,obs,b.par,b.config["control_dt"],2 if name=="c3bf" else 8)
        Blist.append(B);Clist.append(C-m-.5*np.linalg.norm(B,axis=1))
    BB=np.r_[np.concatenate(Blist),np.eye(3),-np.eye(3)]
    CC=np.r_[np.concatenate(Clist),np.full(6,1.5)]
    u,ok,_=SparseProjectionQP(len(CC)).solve(np.zeros(3),BB,CC)
    if not ok:raise RuntimeError("shared_initial_trim_infeasible: scene/seed not altered")
    margin=float(np.min(BB@u+CC));assert margin>=-1e-7
    force=u+[0,0,9.81];b3=force/np.linalg.norm(force);b2=np.cross(b3,[1,0,0]);b2/=np.linalg.norm(b2);b1=np.cross(b2,b3)
    R=np.column_stack([b1,b2,b3]);q=Rotation.from_matrix(R).as_quat()
    omega=np.sqrt(quad_params["mass"]*np.linalg.norm(force)/(4*quad_params["k_eta"]))
    return dict(acceleration=u.tolist(),quaternion_xyzw=q.tolist(),rotor_speeds=[float(omega)]*4,
        minimum_common_initial_margin=margin,definition="Minimum-norm intersection of all four initial sampled/tracking sets; same p/v/obstacles, same physical trim for all.")
