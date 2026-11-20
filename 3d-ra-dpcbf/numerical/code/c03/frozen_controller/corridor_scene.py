"""Load the existing four-obstacle experiment without regenerating its motion."""
from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parent

def load_corridor(mode='bounded'):
    if mode!='bounded':raise ValueError('The frozen corridor is continuous bounded motion; use mode=bounded.')
    path=ROOT/'reference_inputs/corridor_crossing.npz'
    metadata=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['source_sha256']:
        raise ValueError('Frozen corridor file differs from the original displayed scene.')
    with np.load(path,allow_pickle=False) as archive:
        scene={key:archive[key].copy() for key in archive.files}
    if hashlib.sha256(scene['tape'].tobytes()).hexdigest()!=metadata['tape_sha256']:
        raise ValueError('Frozen obstacle states changed.')
    scene.update(tape_dt=float(scene['tape_dt']),mode=int(scene['mode']),revision=int(scene['revision']),
        same_reference_knots=True,source='reference_inputs/corridor_crossing.npz',
        geometry_seed_source='Frozen corridor revision 4; manoeuvre seed is not used.',manoeuvre_seed=None)
    return scene
