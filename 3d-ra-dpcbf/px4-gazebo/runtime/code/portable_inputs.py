"""File/import/CPU validation only; never launches PX4 or computes control."""
from pathlib import Path
import os,json,importlib.util,sys
ROOT=Path(__file__).resolve().parents[1]

def control_core(root=None,allowed=None,require_background=False):
    text=os.environ.get('CBF_CONTROL_CPU')
    if text is None:raise ValueError('Specify a local CPU with --cpu or CBF_CONTROL_CPU; archived CPU benchmarks are not local configuration')
    core=int(text)
    if not hasattr(os,'sched_getaffinity'):raise RuntimeError('PX4 execution requires Linux sched_getaffinity')
    allowed=set(os.sched_getaffinity(0) if allowed is None else allowed)
    if core not in allowed:raise ValueError('CPU is outside current affinity: '+str(sorted(allowed)))
    sibling_path=Path('/sys/devices/system/cpu')/('cpu'+str(core))/'topology/thread_siblings_list'
    from affinity_isolation77 import parse_mask
    sibling=parse_mask(sibling_path.read_text())
    if require_background and not allowed-sibling:raise ValueError('No separate CPU remains for owned background tasks')
    return core

def preflight(args):
    errors=[]
    if sys.platform!='linux':errors.append('PX4/Gazebo requires Linux; this check does not launch it')
    if Path(args.name).name!=args.name or args.name in ('','.','..') or '/' in args.name or '\\' in args.name:errors.append('name must be a single new run-directory name')
    expected_partition='cbf_compare_'+args.name
    if os.environ.get('GZ_PARTITION',expected_partition)!=expected_partition:errors.append('GZ_PARTITION must match this unique run name: '+expected_partition)
    out=ROOT/'runs'/args.name
    if out.exists():errors.append('Existing output preserved: '+str(out))
    if args.synchronous:errors.append('Continuous runner forbids world stepping')
    if args.test!='response':
        if args.scene is None and args.test not in ('four','single'):errors.append('Supply --scene for ten/fifty/hundred; no automatic scene substitution')
        args.scene=(args.scene or ROOT/'scenes/four_obstacle_scene.npz').expanduser().resolve()
        if not args.scene.is_file():errors.append('Missing scene: '+str(args.scene))
        else:
            import numpy as np
            with np.load(args.scene,allow_pickle=False) as z:
                for key in ('tape','tape_dt','start','goal'):
                    if key not in z.files:errors.append('Missing scene array: '+key)
                if 'tape' in z.files and 'tape_dt' in z.files and args.test!='single' and args.duration>(len(z['tape'])-1)*float(z['tape_dt'])+1e-9:errors.append('Duration exceeds frozen scene tape')
    elif args.native_view:errors.append('response mode does not have obstacle display input; omit --native-view')
    if args.initialize:
        args.initialization_file=(args.initialization_file or ROOT/'common_initialization.json').expanduser().resolve()
        if not args.initialization_file.is_file():errors.append('Missing initialization; supply --initialization-file: '+str(args.initialization_file))
        else:
            value=json.loads(args.initialization_file.read_text('utf8'))
            if len(value.get('command',[]))!=3 or not float(value.get('duration_s',0))>0:errors.append('Invalid original initialization file')
    if args.cpu is not None:os.environ['CBF_CONTROL_CPU']=str(args.cpu)
    try:core=control_core(require_background=True)
    except (ValueError,RuntimeError,OSError) as e:core=None;errors.append(str(e))
    for module in ('numpy','scipy','numba','osqp','clarabel','pymavlink'):
        if importlib.util.find_spec(module) is None:errors.append('Missing Python module: '+module)
    system_python=Path('/usr/lib/python3/dist-packages')
    if system_python.is_dir() and str(system_python) not in sys.path:sys.path.append(str(system_python))
    for module in ('gz.transport13','gz.msgs10'):
        try:
            if importlib.util.find_spec(module) is None:errors.append('Missing '+module)
        except ModuleNotFoundError:errors.append('Missing '+module)
    from platform_session import FIRMWARE,PX4,PY
    for p in (FIRMWARE,Path(PY),PX4/'Tools/simulation/gz/worlds/default.sdf',PX4/'src/modules/simulation/gz_bridge/server.config',ROOT/'build/libra_obstacle_motion.so',ROOT/'build/libperiodic_publish75.so'):
        if not p.is_file():errors.append('Missing platform input: '+str(p))
    return {'passed':not errors,'no_simulation_started':True,'runtime':str(ROOT),'scene':str(args.scene) if args.scene else None,'out':str(out),'local_cpu':core,'errors':errors}
