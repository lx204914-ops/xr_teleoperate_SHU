"""Read only a precomputed 3D scene tape; the controller sees current states."""
from pathlib import Path
import os,hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load_reselection(mode='bounded'):
    if mode!='bounded':raise ValueError('This scene requires continuous bounded motion.')
    path=Path(os.environ.get('RA_DPCBF_SCENE_NPZ',str(ROOT/'current_method/reference_inputs/reselection3d.npz'))).resolve()
    if not path.is_relative_to(ROOT):raise ValueError('Scene inputs must be inside the F project.')
    with np.load(path,allow_pickle=False) as archive:a={key:archive[key].copy() for key in archive.files}
    a.update(tape_dt=float(a['tape_dt']),mode=int(a['mode']),same_reference_knots=True,
             source=str(path.relative_to(ROOT)),geometry_seed_source='Offline independent C2 3D manoeuvre tape.',
             manoeuvre_seed=int(a['seed']),scene_file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return a
