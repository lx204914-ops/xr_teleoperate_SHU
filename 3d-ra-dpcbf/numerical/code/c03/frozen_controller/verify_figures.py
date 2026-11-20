"""Decode every GIF and check all published source links and surface semantics."""
from pathlib import Path
import json,hashlib,re
from urllib.parse import urlsplit
from PIL import Image
from result_paths import result_root
ROOT=Path(__file__).resolve().parent
OUT=result_root()

def verify():
    index=json.loads((OUT/'scene_index.json').read_text(encoding='utf8'));rows=[]
    for scene in index:
        folder=OUT/scene['path'];images=folder/'figures'
        audit=json.loads((images/'animation_audit.json').read_text(encoding='utf8'))
        assert audit['surface_is_2D'] and not audit['parabolic_section_lines']
        assert audit['full_angular_extent_rad']>6.28 and audit['max_boundary_equation_residual']<1e-10
        assert audit['source_history_sha256']==hashlib.sha256((folder/'history.npz').read_bytes()).hexdigest()
        with Image.open(images/'trajectory.gif') as im:
            frames=im.n_frames;size=list(im.size);first=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
            for k in range(frames):im.seek(k);im.load()
            last=hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()
        assert frames==len(audit['rendered_source_indices']) and first!=last
        for name in ['trajectory_summary.png','trajectory_summary.pdf','animation_preview.png','control_diagnostics.png',
                     'control_diagnostics.pdf','avoidance_sequence.png','trajectory_projections.png']:
            if name.endswith('.png'):
                with Image.open(images/name) as im:im.verify()
            else:assert (images/name).read_bytes().startswith(b'%PDF-')
        gallery=list(images.glob('all_obstacle_surfaces_*.png'))
        assert len(gallery)==(scene['obstacles']+9)//10
        rows.append(dict(scene=scene['scene'],mode=scene['mode'],frames=frames,size=size,complete_velocity_surfaces=True,
                         source_hash_verified=True,full_gif_decoded=True,gallery_pages=len(gallery)))
    text=(OUT/'index.html').read_text(encoding='utf8');links=re.findall(r'(?:href|src)="([^"]+)"',text)
    missing=[link for link in links if not (OUT/urlsplit(link).path).exists()]
    assert not missing,missing
    for file in OUT.rglob('*.gif'):
        with Image.open(file) as im:
            assert im.n_frames>1
            for k in range(im.n_frames):im.seek(k);im.load()
    result=dict(checked_main_GIFs=len(rows),gallery_links_valid=True,all_22_scenes_exported=len(rows)==22,scenes=rows)
    (OUT/'figure_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps({k:result[k] for k in result if k!='scenes'}))

if __name__=='__main__':verify()
