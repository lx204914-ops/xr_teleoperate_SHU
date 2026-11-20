"""Explicit archived generator binding: only the outer attempt budget is 10000."""
from pathlib import Path
import argparse,csv,sys

ROOT=Path(__file__).resolve().parent

def main():
    p=argparse.ArgumentParser(description='只在显式请求时生成一个正式场景，并核对历史tape哈希和接受序号。')
    p.add_argument('--count',type=int,choices=[1,10,30,50,100],default=10)
    p.add_argument('--radius',type=float,choices=[.3,.5,.7],default=.3)
    p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--out',type=Path,default=ROOT/'outputs/generated_scene.npz')
    p.add_argument('--generate',action='store_true')
    args=p.parse_args()
    if not args.generate:p.print_help();return
    if args.out.exists() or args.out.with_suffix('.json').exists():raise FileExistsError('拒绝覆盖已有场景。')
    base=ROOT/'code/conditional_6000'
    sys.path.insert(0,str(base/'current_method_conditional'));sys.path.insert(0,str(base/'code'))
    from generate_scenes import build_scene,save_scene
    scene=build_scene(args.count,args.radius,args.seed,max_attempts=10000)
    with (base/'inputs/source_scene_hashes.csv').open(encoding='utf-8-sig',newline='') as f:
        ref=next(r for r in csv.DictReader(f) if r['scene_id']==scene['scene_id'])
    assert scene['tape_sha256']==ref['scene_tape_sha256']
    assert int(scene['metadata']['accepted_attempt'])==int(ref['generator_attempt'])
    save_scene(scene,args.out)
    print('场景tape与接受序号核验通过：'+scene['scene_id'])

if __name__=='__main__':main()
