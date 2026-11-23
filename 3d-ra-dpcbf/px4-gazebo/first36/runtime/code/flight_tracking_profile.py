"""Simulation-only PX4 execution tuning, verified readback and restoration."""
from pathlib import Path
import json,time,struct
PROFILE={'MC_ROLL_P': 12.0, 'MC_PITCH_P': 12.0, 'THR_MDL_FAC': 0.7391304347826086, 'MPC_THR_HOVER': 0.5826357889112165, 'MPC_ACC_DECOUPLE': 0}

_profile_file=Path(__file__).resolve().parents[1]/'execution_profile.json'
if _profile_file.exists():
 PROFILE.update(json.loads(_profile_file.read_text())['parameters'])

def exchange(link,name,value=None):
 ident=name.encode();deadline=time.monotonic()+4;last=0.
 if not hasattr(link,'_cbf_param_types'):link._cbf_param_types={}
 while time.monotonic()<deadline:
  if time.monotonic()-last>.3:
   if value is None:link.m.mav.param_request_read_send(link.m.target_system,link.m.target_component,ident,-1)
   else:
    kind=link._cbf_param_types.get(name,9)
    wire=struct.unpack('<f',struct.pack('<i',int(value)))[0] if kind==6 else float(value)
    link.m.mav.param_set_send(link.m.target_system,link.m.target_component,ident,wire,kind)
   last=time.monotonic()
  msg=link.m.recv_match(type='PARAM_VALUE',blocking=True,timeout=.1)
  if msg is not None and str(msg.param_id).strip('\x00')==name:
   kind=int(msg.param_type);link._cbf_param_types[name]=kind
   decoded=float(struct.unpack('<i',struct.pack('<f',msg.param_value))[0]) if kind==6 else float(msg.param_value)
   if value is None or abs(decoded-value)<1e-5:return decoded
 raise RuntimeError('PX4 parameter readback failed: '+name)

def apply(link,out):
 record=dict(SITL_only=True,original={},requested=PROFILE,applied={},restored=False,
  rationale='Acceleration-only offboard avoids the additional velocity-feedback term; SITL attitude P 10 to 12 to test measured attitude lag, within official parameter range [0,12]. No safety output postfilter.')
 for key,value in PROFILE.items():
  record['original'][key]=exchange(link,key);record['applied'][key]=exchange(link,key,value)
 (Path(out)/'flight_tracking_profile.json').write_text(json.dumps(record,indent=2));return record

def restore(link,out,record):
 errors=[]
 for key,value in record['original'].items():
  try:exchange(link,key,value)
  except Exception as exc:errors.append(str(exc))
 record.update(restored=not errors,restore_errors=errors)
 (Path(out)/'flight_tracking_profile.json').write_text(json.dumps(record,indent=2))
