"""Offline file/data/AST verification only. Never import scientific runtimes."""
from pathlib import Path
import ast,csv,hashlib,json,re,sys
ROOT=Path(__file__).resolve().parents[1]

def rows(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    checks=rows(ROOT/'reproducibility/发布文件校验.csv')
    for row in checks:
        p=ROOT/row['文件']
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['SHA256']:
            raise AssertionError('文件校验失败：'+row['文件'])
    code=list(ROOT.rglob('*.py'))
    for p in code:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
    data=rows(ROOT/'numerical/data/formal_6000.csv')
    assert len(data)==len({(r['scene_id'],r['method']) for r in data})==6000
    assert len({r['scene_id'] for r in data})==1500
    expected={'ra_conditional_detour':1410,'park_3d':1375,'ma_cbf_vo':1345,'c3bf':865}
    for method,success in expected.items():
        part=[r for r in data if r['method']==method]
        assert len(part)==1500 and sum(r['outcome_class']=='goal' for r in part)==success
    assert len(rows(ROOT/'numerical/data/60单元汇总.csv'))==60
    bound=rows(ROOT/'rotorpy/data/六次轨迹来源绑定.csv')
    assert len(bound)==6
    for row in bound:
        p=ROOT/'rotorpy'/row['发布轨迹']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['轨迹SHA256']
    four={m:json.loads((ROOT/f'rotorpy/data/four_{m}_summary.json').read_text(encoding='utf8')) for m in ['ra','park','ma_cbf_vo','c3bf']}
    assert [round(four[m]['first_arrival_s'],3) for m in ['ra','park','ma_cbf_vo','c3bf']]==[14.33,12.97,9.655,11.1]
    runs=json.loads((ROOT/'px4-gazebo/data/results.json').read_text(encoding='utf8'))['runs']
    assert len(runs)==len({r['session'] for r in runs})==100
    for method,n,k in [('ra',22,16),('park',12,7),('ma_cbf_vo',15,0),('c3bf',0,0)]:
        group=[r for r in runs if r['method']==method]
        assert len(group)==25 and sum(r['arrived'] for r in group)==n and sum(r['strict_execution_passed'] for r in group)==k
    broken=[]
    for p in ROOT.rglob('*.md'):
        text=p.read_text(encoding='utf8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            target=target.split('#')[0]
            if not target or '://' in target or target.startswith('mailto:'):continue
            if not (p.parent/target).exists():broken.append((str(p.relative_to(ROOT)),target))
    assert not broken,broken
    print(json.dumps({'核验':'通过','文件SHA数量':len(checks),'Python语法文件数':len(code),
      '正式逐次记录':len(data),'PX4逐次记录':len(runs),'本地文档断链数':len(broken),
      '边界':'离线核验，不代表科学控制重跑、Linux SITL复现或真机验证通过'},ensure_ascii=False,indent=2))

if __name__=='__main__':main()
