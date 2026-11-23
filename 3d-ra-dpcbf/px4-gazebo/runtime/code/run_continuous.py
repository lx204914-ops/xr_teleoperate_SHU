"""Portable CLI; --help/--check-only never import the flight engine."""
from pathlib import Path
import argparse,json,os

def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--cruise', type=float, default=1.0)
    ap.add_argument('--native-view', action='store_true')
    ap.add_argument('--wall-rtf', type=float, default=1.0)
    ap.add_argument('--initialize', action='store_true')
    ap.add_argument('--method', choices=['ra', 'ma_cbf_vo', 'c3bf', 'park'], default='ra')
    ap.add_argument('--test', choices=['response', 'single', 'four', 'ten', 'fifty', 'hundred'], default='response')
    ap.add_argument('--scene', type=Path)
    ap.add_argument('--duration', type=float, default=18.0)
    ap.add_argument('--execution', choices=['acceleration', 'velocity_ff'], default='acceleration')
    ap.add_argument('--synchronous', action='store_true')
    ap.add_argument('--gui', action='store_true')
    ap.add_argument('--cpu',type=int,help='本机可用的控制CPU核；不会沿用历史机器核号')
    ap.add_argument('--initialization-file',type=Path,help='仅--initialize时使用的原初始化文件')
    ap.add_argument('--check-only',action='store_true',help='只检查依赖和输入；绝不启动SITL')
    return ap

def main():
    args=build_parser().parse_args()
    from portable_inputs import preflight
    report=preflight(args)
    if args.check_only:
        print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['passed'] else 2
    if not report['passed']:raise RuntimeError('Startup preflight failed before SITL: '+ '; '.join(report['errors']))
    os.environ.update(GZ_PARTITION='cbf_compare_'+args.name,GZ_IP='127.0.0.1',CBF_COMPARE_METHOD=args.method,CBF_METHOD_LABEL={'ra':'RA-DPCBF','park':'Park-inspired','ma_cbf_vo':'MA-CBF-VO','c3bf':'C3BF'}[args.method])
    import _run_continuous_frozen as engine
    engine.main(args)
    return 0

if __name__=='__main__':raise SystemExit(main())
