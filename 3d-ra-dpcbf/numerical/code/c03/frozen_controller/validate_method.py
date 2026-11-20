"""Validate source/parameter bindings and unit tests, never run navigation scenes."""
from pathlib import Path
import argparse,ast,hashlib,json,re,subprocess,sys,time
import xml.etree.ElementTree as ET
from configuration import load_config,config_sha256,core_config,output_root

ROOT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir',default='qa/method_completion_20261018')
    args=parser.parse_args();dest=output_root(args.report_dir);dest.mkdir(parents=True,exist_ok=True)
    cfg=load_config();cc=core_config(cfg)
    manifest=json.loads((ROOT/'upstream_import.json').read_text(encoding='utf8'))
    original={r['path'].replace('\\','/'):r['source_sha256'] for r in manifest['files']}
    preserved=['ra_dpcbf/core.py','dynamic_study/kernel.py','nominal_mpc.py','sampled_safety.py']
    same={r:hashlib.sha256((ROOT/r).read_bytes()).hexdigest()==original[r] for r in preserved}
    sources=[]
    for f in ROOT.rglob('*.py'):
        if any(s in ('__pycache__','.pytest_cache') for s in f.parts):continue
        ast.parse(f.read_text(encoding='utf-8-sig'),filename=str(f))
        sources.append({'path':f.relative_to(ROOT).as_posix(),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    # Test a copy of the code in place; no benchmark command is invoked.
    started=time.perf_counter();junit=dest/'method_tests.xml'
    command=[sys.executable,'-B','-m','pytest',str(ROOT/'tests'),'-q',
             '--junitxml='+str(junit),'-o','cache_dir='+str(dest/'pytest_cache')]
    result=subprocess.run(command,cwd=ROOT,text=True,encoding='utf8',errors='replace',capture_output=True)
    combined=result.stdout+result.stderr
    (dest/'test_output.txt').write_text(combined,encoding='utf8')
    counts=dict(tests=0,failures=0,errors=0,skipped=0)
    if junit.exists():
        for suite in ET.parse(junit).getroot().iter('testsuite'):
            for key in counts:counts[key]+=int(suite.get(key,0))
    warning_match=re.search(r'(\d+) warnings? in',combined)
    warning_count=int(warning_match.group(1)) if warning_match else 0
    warning_lines=sorted({line.strip() for line in combined.splitlines() if 'Warning:' in line})
    expected_warning_tokens=[
        'DeprecationWarning: "polish" is deprecated.',
        'PendingDeprecationWarning: The default value of raise_error will change to True',
        'UserWarning: Solution may be inaccurate.',
    ]
    warning_cases=['test_hard_qp_and_joint_infeasibility','test_sampling_margin_reaches_generic_qp']
    reviewed_warnings=(warning_count==0 or (warning_count==6 and
        all(any(token in line for line in warning_lines) for token in expected_warning_tokens) and
        all(case in combined for case in warning_cases) and
        all(any(token in line for token in expected_warning_tokens) for line in warning_lines)))
    warning_review=dict(count=warning_count,lines=warning_lines,cases=warning_cases,
        reviewed_nonblocking=reviewed_warnings,
        disposition=('No warnings.' if warning_count==0 else
            'CVXPY 1.4.3/OSQP 1.1.1 interface deprecations plus solver inaccuracy warnings '
            'occur only in two deliberate feasible/infeasible boundary tests. SafetyQP accepts a '
            'solution only after finite-value, input-box and hard-constraint residual checks at 1e-7; '
            'the tests also require residual bounds and rejection of infeasible instances.'))
    row=dict(passed=result.returncode==0 and all(same.values()) and reviewed_warnings,tests=counts,
        warning_review=warning_review,
        method_config_sha256=config_sha256(cfg),unchanged_upstream_sources=same,
        source_manifest=sources,python=sys.executable,runtime_s=time.perf_counter()-started,
        active_method_dir=str(ROOT),lifecycle_rule=cfg['lifecycle_rule'],
        numerical_defaults_match_active_parameters=bool(cc.clearance_time==cfg['parameters'][0]),
        navigation_experiments_run=False,statistical_experiments_run=False,figures_generated=False,
        scope='Code, formula, optimization and lifecycle regression tests; no claim of navigation success or real-time feasibility.')
    (dest/'validation.json').write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding='utf8')
    print(result.stdout)
    print(json.dumps({k:row[k] for k in ('passed','tests','unchanged_upstream_sources','active_method_dir')},ensure_ascii=False,indent=2))
    raise SystemExit(0 if row['passed'] else 1)

if __name__=='__main__':main()
