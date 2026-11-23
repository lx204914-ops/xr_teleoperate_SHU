"""Verify native screenshot metadata against every saved render ACK."""
from pathlib import Path
import json
import hashlib
import numpy as np
from PIL import Image, ImageSequence


def audit(root):
    root = Path(root)
    folder = root / 'native_recording'
    meta = json.loads((folder / 'recording_receipt.json').read_text())
    run = root / 'runs' / meta['session']
    with np.load(run / 'native_frame_audit.npz', allow_pickle=False) as z:
        frames = z['frames']
    ids = frames[:, 2].astype(int)
    assert np.array_equal(ids, np.arange(1, len(ids) + 1))
    lookup = {int(row[2]): row for row in frames}
    for record in meta['records']:
        before = lookup[record['native_frame_before']]
        after = lookup[record['native_frame_after']]
        assert before[11] == after[11] == 0
        assert abs(before[1] - record['sim_before_s']) < 1e-10
        assert abs(after[1] - record['sim_after_s']) < 1e-10
        assert record['native_frame_after'] >= record['native_frame_before']
        n = int(before[6])
        params = before[12:].reshape(n, 33)
        assert n in (50, 100)
        assert np.max(abs(params[:, 5] - np.array(record['curvature_vertex'])[:, 0])) < 1e-12
        assert np.max(abs(params[:, 4] - np.array(record['curvature_vertex'])[:, 1])) < 1e-12
        assert np.max(abs(params[:, 6] - record['risk_radius'])) < 1e-12
        assert abs(params[0, 9] - record['controller_stamp_before']) < 1e-10
    gif = folder / 'gazebo_native_live.gif'
    unique = set()
    duration = 0
    with Image.open(gif) as im:
        count = im.n_frames
        for image in ImageSequence.Iterator(im):
            unique.add(hashlib.sha256(image.convert('RGB').tobytes()).hexdigest())
            duration += image.info.get('duration', 0)
    assert count == len(meta['records']) == meta['frames']
    assert len(unique) == count
    assert abs(duration / 1000 - meta['playback_duration_s']) < .011
    assert hashlib.sha256(gif.read_bytes()).hexdigest() == meta['sha256']
    result = dict(passed=True, run=meta['session'], native_render_frames=len(ids),
                  missing_render_ACKs=0, GIF_frames=count, unique_GIF_frames=len(unique),
                  GIF_frames_with_before_and_after_geometry_audits=count,
                  duration_s=duration / 1000, sha256=meta['sha256'],
                  scope='Screenshots are bracketed by real native render ACKs; no claim of pixel capture atomicity.')
    (folder / 'frame_coverage_check.json').write_text(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    print(json.dumps(audit(Path(__file__).resolve().parents[1]), indent=2))
