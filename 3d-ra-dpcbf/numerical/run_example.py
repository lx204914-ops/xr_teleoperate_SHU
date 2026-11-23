"""Explicit single-case launcher for the frozen conditional controller/baselines.

Default invocation only prints help. No batch scheduler, data overwrite or auto-resume.
The wrapper changes only scene/output binding, not controller equations or parameters.
"""
from pathlib import Path
import argparse,json,sys,types

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser(description='读取已发布场景，手动运行一个方法；默认不启动仿真。')
    parser.add_argument('--scene',type=Path,default=ROOT/'scenes/ring10_actual.npz')
    parser.add_argument('--method',choices=['ra','park','c3bf','ma_cbf_vo'],default='ra')
    parser.add_argument('--out',type=Path,default=ROOT/'outputs/example')
    parser.add_argument('--run',action='store_true',help='显式开始单例仿真，输出目录必须不存在')
    parser.add_argument('--check-only',action='store_true',help='只检查文件、场景与输出位置，不导入控制器或启动仿真')
    args=parser.parse_args()
    if not args.run and not args.check_only:parser.print_help();return
    args.scene=args.scene.expanduser().resolve(strict=True)
    args.out=args.out.expanduser().resolve()
    if args.out.exists():raise FileExistsError('拒绝覆盖已有输出根目录：'+str(args.out))
    import numpy as np
    base=ROOT/'code/conditional_6000';core=base/'current_method_conditional'
    json.loads((core/'method_config.json').read_text(encoding='utf8'))
    if args.method=='ma_cbf_vo':json.loads((base/'inputs/ma_settings.json').read_text(encoding='utf8'))
    with np.load(args.scene,allow_pickle=False) as z:scene={k:z[k].copy() for k in z.files}
    scene['tape_dt']=float(scene['tape_dt']);scene['mode']=int(scene.get('mode',1))
    if scene['mode']!=1:raise ValueError('单例入口只接受已保存的连续运动tape。')
    scene.setdefault('source',str(args.scene));scene.setdefault('same_reference_knots',False)
    scene.setdefault('waypoints',np.asarray(scene['goal'])[None,:]);scene['scene_id']=args.scene.stem
    variant={'ra':'conditional_detour','park':'park_3d','c3bf':'c3bf','ma_cbf_vo':'ma_cbf_vo'}[args.method]
    destination=args.out/'bounded'/scene['scene_id']/variant
    if args.check_only:
        print(json.dumps({'检查':'通过','未启动仿真':True,'场景':str(args.scene),'输出根目录':str(args.out),'实际结果目录':str(destination)},ensure_ascii=False));return
    for p in [core,base/'code',ROOT/'code/supplementary_adapters']:sys.path.insert(0,str(p))
    boot=types.ModuleType('bootstrap');boot.MAIN=core;boot.HERE=base;boot.LABELS={
      'ra_conditional_detour':'RA-DPCBF','park_3d':'Park适配','c3bf':'C3BF','ma_cbf_vo':'MA-CBF-VO'}
    sys.modules['bootstrap']=boot
    # Reserve one fresh root atomically; all four backends use the same layout.
    args.out.mkdir(parents=True,exist_ok=False)
    if args.method=='ra':
        import run_suite
        run_suite.load_scene=lambda *a,**k:scene
        horizon=(len(scene['tape'])-1)*scene['tape_dt']
        run_suite.run(scene['scene_id'],'bounded','conditional_detour',seed=17,duration=horizon,out=str(args.out),save_history=True)
    elif args.method=='ma_cbf_vo':
        from ma_runner import ma_baseline
        ma_baseline(scene,'ma_cbf_vo',destination,save_history=True)
    else:
        from baseline_runner import baseline
        baseline(scene,'park_3d' if args.method=='park' else 'c3bf',destination,save_history=True)

if __name__=='__main__':main()
