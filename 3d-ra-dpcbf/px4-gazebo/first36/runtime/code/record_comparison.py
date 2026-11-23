from pathlib import Path
import sys,json,os,time,traceback
import record_continuous_gif as original
ROOT=Path(__file__).resolve().parents[1]
try:original.main()
except Exception as exc:
 name=sys.argv[sys.argv.index('--attach-session')+1];out=ROOT/'native_recording'/name;run=ROOT/'runs'/name
 deadline=time.monotonic()+30
 while not (run/'summary.json').exists() and time.monotonic()<deadline:time.sleep(.2)
 summary=json.loads((run/'summary.json').read_text()) if (run/'summary.json').exists() else {}
 out.mkdir(parents=True,exist_ok=True)
 try:original.screenshot().save(out/'native_terminal.png');captured=True
 except Exception:captured=False
 phase='completed_short_failure_native_PNG' if not summary.get('completed') else 'completed_without_recording'
 (out/'recording_status.json').write_text(json.dumps(dict(session=name,phase=phase,native_terminal_pixel_capture=captured,recording_error=str(exc),successful_GIF=False),indent=2))
 print('REAL_RECORDING_DIAGNOSTIC',str(exc),flush=True)
