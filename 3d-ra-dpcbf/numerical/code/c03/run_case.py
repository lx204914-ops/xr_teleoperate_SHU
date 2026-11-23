"""Manual C03 entry. Reuse frozen bindings; never search or auto-resume."""
from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parent
METHODS=('full','no_detour','no_reselection','no_progress_guard','fixed_radius','no_transverse','fixed_strength')

def main():
    parser=argparse.ArgumentParser(description='已保存C03单例；默认不运行，不搜索新场景。')
    parser.add_argument('--method',choices=METHODS,default='full')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/manual')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if not args.run and not args.check_only:parser.print_help();return
    out=args.out.expanduser().resolve()
    if out.exists():raise FileExistsError('Existing output root preserved: '+str(out))
    sys.path.insert(0,str(ROOT/'code'))
    import bootstrap
    bootstrap.verify()
    from bindings47 import bindings
    from no_reselection import paused_direction_transport
    import evaluation_run as runtime
    import numpy as np
    sid='tetra12_C03_S8203';p=ROOT/'inputs/scenes'/(sid+'.npz')
    metadata=json.loads(p.with_suffix('.json').read_text('utf8'))
    with np.load(p,allow_pickle=False) as z:scene={k:z[k].copy() for k in z.files}
    scene.update(tape_dt=float(scene['tape_dt']),mode=int(scene['mode']),scene_id=sid,metadata=metadata,same_reference_knots=True,manoeuvre_seed=metadata['seed'],geometry_seed_source=str(p))
    cfg,factory,variant=bindings(args.method)
    if args.check_only:
        print(json.dumps({'检查':'通过','未启动仿真':True,'场景':sid,'方法':args.method,'原时长/s':40.,'实际输出':str(out/'bounded'/sid/variant)},ensure_ascii=False));return
    runtime.load_scene=lambda *a,**k:scene
    runtime.load_config=lambda *a,**k:json.loads(json.dumps(cfg))
    runtime.SafetyFilter=factory
    # Same paused-direction transport binding as the original C03 executor.
    advance_original=runtime.advance
    if args.method=='no_reselection':
        def advance(*a):
            mech=factory.current.mechanism;paused=mech.lock_j is not None and a[13]<0
            if paused:p0=a[0][:3].copy();v=a[0][3:].copy();d=mech.lock_d.copy();e=mech.lock_e.copy()
            value=advance_original(*a)
            if paused:
                d,e=paused_direction_transport(p0,v,a[4],a[7],a[8],a[9],a[10],a[11],a[12],mech.lock_j,d,e)
                mech.lock_d=d;mech.lock_e=e;mech.d=d.copy();mech.e=e.copy()
            return value
        runtime.advance=advance
    out.mkdir(parents=True,exist_ok=False)
    runtime.run(sid,'bounded',variant,seed=metadata['seed'],duration=40.,out=str(out),save_history=True)

if __name__=='__main__':main()
