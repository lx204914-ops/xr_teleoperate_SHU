"""Unpack the archived SITL binary only. Never execute firmware or start PX4."""
from pathlib import Path
import argparse,gzip,hashlib,os
ROOT=Path(__file__).resolve().parent
EXPECTED='eeb37b2708909381f88bf56de1c00d69dac7eb0dc8640074cee6c357a83cf67a'

def main():
    parser=argparse.ArgumentParser(description='只解包原SITL固件，已有文件拒绝覆盖，不运行。')
    parser.add_argument('--runtime',choices=('first36','later100'),required=True)
    args=parser.parse_args()
    runtime=ROOT/('first36/runtime' if args.runtime=='first36' else 'runtime')
    target=runtime/'build/feedback_receipt_bin/px4'
    if target.exists():raise FileExistsError('已有固件保留：'+str(target))
    data=gzip.decompress((ROOT/'artifacts/px4_sitl_linux_x86_64.gz').read_bytes())
    if hashlib.sha256(data).hexdigest()!=EXPECTED:raise ValueError('原固件SHA256不匹配')
    target.parent.mkdir(parents=True,exist_ok=True)
    try:
        with target.open('xb') as f:f.write(data)
    except FileExistsError:raise
    except BaseException:
        target.unlink(missing_ok=True);raise
    os.chmod(target,0o755)
    print('已解包并核对SHA256，未执行：'+str(target))

if __name__=='__main__':main()
