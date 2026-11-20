"""Standalone RA physical verification; no four-method comparison at this density."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent;ROOT=PACKAGE.parent
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.vehicles.crazyflie_params import quad_params
from deployed_controller import DeployedController
from sparse_projection import SparseProjectionQP
import run_quadrotor
def ra_trim(initial,goal,obs):
    c=DeployedController();c.reset(initial,goal,obs);c.execution_acceleration=np.zeros(3)
    _,status,d=c.step(0.,initial,obs)
    if "B" not in d:raise RuntimeError("RA initial domain failure: "+status)
    B=np.r_[d["B"],np.eye(3),-np.eye(3)]
    C=np.r_[d["C"]-d["sampling_margin"],np.full(6,1.5)]
    u,ok,_=SparseProjectionQP(len(C)).solve(np.zeros(3),B,C)
    if not ok:raise RuntimeError("RA own initial robust sampled set is empty")
    assert np.min(B@u+C)>=-1e-7
    force=u+[0.,0.,9.81];b3=force/np.linalg.norm(force)
    b2=np.cross(b3,[1.,0.,0.]);b2/=np.linalg.norm(b2);b1=np.cross(b2,b3)
    q=Rotation.from_matrix(np.column_stack([b1,b2,b3])).as_quat()
    rotor=np.sqrt(quad_params["mass"]*np.linalg.norm(force)/(4*quad_params["k_eta"]))
    return dict(acceleration=u.tolist(),quaternion_xyzw=q.tolist(),rotor_speeds=[float(rotor)]*4,
       minimum_common_initial_margin=float(np.min(B@u+C)),
       definition="Standalone RA validation: minimum-norm RA initial sampled/tracking feasible acceleration. No baseline intersection; not a four-method comparison.")
if __name__=="__main__":
    scene=ROOT/"tracked_interface/inputs/smooth_N050_S2000.npz"
    assert hashlib.sha256(scene.read_bytes()).hexdigest()=="00ba192a5a4f06fb62adc3fe745a527e86aac041c8e0949ef773399d5fdc425a"
    # Only platform initialization and output root differ. Original functions,
    # flight controller, execution limits, scene and obstacle tape are unchanged.
    run_quadrotor.common_trim=ra_trim
    run_quadrotor.__file__=__file__
    run_quadrotor.run(scene,horizon=24.,method="ra")
