"""Infrastructure regression tests. Fake backends/processes only; no simulation."""
from pathlib import Path
from unittest import mock
import contextlib,importlib.util,io,json,os,sys,tempfile,types,unittest
ROOT=Path(__file__).resolve().parents[1]

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

class EntryTests(unittest.TestCase):
    def test_four_backends_share_absolute_output_layout(self):
        entry=load('portable_example',ROOT/'numerical/run_example.py')
        with tempfile.TemporaryDirectory() as temp:
            cwd=Path.cwd();os.chdir(temp)
            try:
                for method,variant in [('ra','conditional_detour'),('park','park_3d'),('c3bf','c3bf'),('ma_cbf_vo','ma_cbf_vo')]:
                    calls=[]
                    ra=types.ModuleType('run_suite')
                    def run(scene,mode,variant,**kw):calls.append(Path(kw['out'])/mode/scene/variant)
                    ra.run=run
                    ma=types.ModuleType('ma_runner');ma.ma_baseline=lambda scene,name,dest,**kw:calls.append(dest)
                    baseline=types.ModuleType('baseline_runner');baseline.baseline=lambda scene,name,dest,**kw:calls.append(dest)
                    argv=['run_example.py','--method',method,'--out','outputs/'+method,'--run']
                    with mock.patch.dict(sys.modules,{'run_suite':ra,'ma_runner':ma,'baseline_runner':baseline}),mock.patch.object(sys,'argv',argv):entry.main()
                    expected=Path(temp).resolve()/'outputs'/method/'bounded/ring10_actual'/variant
                    self.assertEqual(calls,[expected]);self.assertTrue(calls[0].is_absolute())
                    sentinel=Path(temp)/'outputs'/method/'previous.txt';sentinel.write_text('original')
                    with mock.patch.object(sys,'argv',argv),self.assertRaises(FileExistsError):entry.main()
                    self.assertEqual(sentinel.read_text(),'original')
            finally:os.chdir(cwd)

    def test_check_only_does_not_import_controller_or_create_output(self):
        entry=load('portable_check',ROOT/'numerical/run_example.py')
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'never_created'
            with mock.patch.object(sys,'argv',['run_example.py','--method','ma_cbf_vo','--out',str(out),'--check-only']),contextlib.redirect_stdout(io.StringIO()) as result:entry.main()
            self.assertFalse(out.exists());self.assertTrue(json.loads(result.getvalue())['未启动仿真'])

class PlatformTests(unittest.TestCase):
    def exercise(self,rel,kind):
        mod=load('platform_under_test',ROOT/rel/'code/platform_session.py')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'runtime';px4=Path(temp)/'px4';py=Path(temp)/'python';py.write_text('not executed')
            worlddir=px4/'Tools/simulation/gz/worlds';worlddir.mkdir(parents=True)
            (worlddir/'default.sdf').write_text('<sdf><world name="default"></world></sdf>')
            config=px4/'src/modules/simulation/gz_bridge/server.config';config.parent.mkdir(parents=True);config.write_text('<server_config><plugins/></server_config>')
            firmware=root/'build/feedback_receipt_bin/px4';firmware.parent.mkdir(parents=True);firmware.write_bytes(b'never executed');firmware.chmod(0o755)
            (root/'build/libra_obstacle_motion.so').write_bytes(b'never loaded')
            (root/'build/libperiodic_publish75.so').write_bytes(b'never loaded')
            logger=px4/'build/px4_sitl_default/rootfs/etc/logging/logger_topics.txt';logger.parent.mkdir(parents=True);logger.write_bytes(b'original logger\n')
            original_world=root/'worlds/ra_lab.sdf';original_world.parent.mkdir();original_world.write_text('original private world')
            link=worlddir/'ra_lab.sdf';links={}
            # Windows runners may lack symlink privilege. Model only this private
            # link; all real file bytes are still checked in a temporary tree.
            real_is=Path.is_symlink;real_unlink=Path.unlink
            def islink(p):return str(p) in links or real_is(p)
            def symlink(p,target,*a,**k):
                self.assertEqual(p,link);p.write_text('mock link');links[str(p)]=str(target)
            def readlink(p):return Path(links[str(p)])
            def unlink(p,*a,**k):links.pop(str(p),None);return real_unlink(p,*a,**k)
            pgroup=[]
            proc=types.SimpleNamespace(pid=99999999,poll=lambda:1,wait=lambda **kw:1,stdin=io.BytesIO())
            with mock.patch.multiple(mod,ROOT=root,PX4=px4,PY=str(py),FIRMWARE=firmware),mock.patch.object(mod.shutil,'which',return_value='mock-command'),mock.patch.object(mod.subprocess,'run',return_value=types.SimpleNamespace(returncode=1)),mock.patch.object(mod.subprocess,'Popen',side_effect=OSError('injected Popen failure') if kind=='popen' else None,return_value=proc) as popen,mock.patch.object(mod.time,'sleep'),mock.patch.object(mod.os,'killpg',side_effect=lambda *a:pgroup.append(a),create=True),mock.patch.object(mod.signal,'SIGKILL',9,create=True),mock.patch.object(Path,'is_symlink',islink),mock.patch.object(Path,'symlink_to',symlink),mock.patch.object(Path,'readlink',readlink),mock.patch.object(Path,'unlink',unlink):
                if kind=='missing':firmware.unlink()
                for i in range(2):
                    with self.assertRaises((OSError,RuntimeError)):
                        with mod.Platform(root/'runs'/('case'+str(i))):self.fail('must not enter flight body')
                    self.assertEqual(logger.read_bytes(),b'original logger\n')
                    self.assertFalse(link.exists());self.assertFalse(links)
                    self.assertEqual(original_world.read_text(),'original private world')
                if kind=='missing':popen.assert_not_called();self.assertFalse((root/'runs').exists())
                for handle in (root/'runs').rglob('px4_gazebo.log') if (root/'runs').exists() else []:
                    with handle.open('ab') as f:f.write(b'closed log check')

    def test_missing_firmware_is_side_effect_free_twice(self):
        for rel in ['px4-gazebo/runtime','px4-gazebo/first36/runtime']:self.exercise(rel,'missing')
    def test_popen_failure_restores_logger_and_link_twice(self):
        for rel in ['px4-gazebo/runtime','px4-gazebo/first36/runtime']:self.exercise(rel,'popen')
    def test_early_process_exit_restores_logger_and_link_twice(self):
        for rel in ['px4-gazebo/runtime','px4-gazebo/first36/runtime']:self.exercise(rel,'exited')

    def test_exit_cleanup_still_runs_when_receipt_writer_fails(self):
        for rel in ['px4-gazebo/runtime','px4-gazebo/first36/runtime']:
            mod=load('exit_under_test',ROOT/rel/'code/platform_session.py')
            platform=mod.Platform('unused_test_path')
            stopped=[];restored=[]
            with mock.patch.object(platform,'_finish',side_effect=OSError('receipt disk error')),mock.patch.object(platform,'_stop_started_process',side_effect=lambda:stopped.append(True)),mock.patch.object(platform,'_restore_local_state',side_effect=lambda:restored.append(True)):
                with self.assertRaises(OSError):platform.__exit__(None,None,None)
            self.assertEqual(stopped,[True]);self.assertEqual(restored,[True])

class PreflightTests(unittest.TestCase):
    def test_missing_scene_and_initialization_stop_before_engine_import(self):
        for rel in ['px4-gazebo/runtime','px4-gazebo/first36/runtime']:
            cli=load('front',ROOT/rel/'code/run_continuous.py')
            inputs=load('inputs_under_test',ROOT/rel/'code/portable_inputs.py')
            with tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                args=cli.build_parser().parse_args(['--name','missing_input','--test','ten','--scene',str(root/'missing.npz'),'--initialize','--initialization-file',str(root/'missing.json')])
                platform=types.SimpleNamespace(FIRMWARE=root/'firmware',PX4=root/'px4',PY=str(root/'python'))
                with mock.patch.object(inputs,'ROOT',root),mock.patch.object(inputs,'control_core',return_value=1),mock.patch.dict(sys.modules,{'platform_session':platform}),mock.patch.object(inputs.importlib.util,'find_spec',return_value=object()):
                    result=inputs.preflight(args)
                self.assertFalse(result['passed']);self.assertTrue(result['no_simulation_started'])
                self.assertTrue(any('Missing scene:' in e for e in result['errors']))
                self.assertTrue(any('Missing initialization;' in e for e in result['errors']))
                self.assertFalse((root/'runs').exists())

    def test_cpu_can_be_read_after_own_thread_is_already_pinned(self):
        inputs=load('cpu_inputs',ROOT/'px4-gazebo/runtime/code/portable_inputs.py')
        affinity=types.SimpleNamespace(parse_mask=lambda s:{2})
        with mock.patch.dict(os.environ,{'CBF_CONTROL_CPU':'2'}),mock.patch.object(inputs.os,'sched_getaffinity',return_value={2},create=True),mock.patch.object(Path,'read_text',return_value='2'),mock.patch.dict(sys.modules,{'affinity_isolation77':affinity}):
            self.assertEqual(inputs.control_core(),2)
            self.assertEqual(inputs.control_core(allowed={2,3},require_background=True),2)
            with self.assertRaises(ValueError):inputs.control_core(require_background=True)

if __name__=='__main__':unittest.main(verbosity=2)
