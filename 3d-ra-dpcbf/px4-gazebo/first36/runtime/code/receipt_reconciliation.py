"""Independent PX4 downstream setpoint confirmation for a truncated ULog tail.

No sent command is treated as received just because it was sent. Require an
actual source-system=1/component=1 PX4 message with the same unique float32
payload, ordered after the last ULog receipt, before physical termination.
Keep the original last-packet exception unchanged and never invent a receipt.
"""
import numpy as np

def reconcile(sent,matched,first,rx_t,payload,trace,summary):
 from accepted_input_receipts75 import supplement
 matched,first,rx_t,payload,accepted=supplement(sent,matched,first,rx_t,payload,trace,summary)
 if summary.get('downstream_payload_is_internal_tracking_reference'):
  return matched,first,rx_t,payload,dict(ULog_only_unmatched=[i for i in range(len(sent)) if i not in first],PX4_downstream_confirmed=[],accepted_input_receipts=accepted,note='Internal feedback-corrected setpoint is not an incoming wire receipt; no substitution is allowed')
 raw_missing=[i for i in range(len(sent)) if i not in first];last=max(first,default=-1)
 wire=np.asarray([v['velocity_enu']+v['acceleration_enu'] for v in sent],dtype=np.float32)
 counts={}
 for row in wire:
  key=tuple(row.tolist());counts[key]=counts.get(key,0)+1
 echoes=[]
 for k,v in enumerate(trace['receive']):
  if v['kind']!='POSITION_TARGET_LOCAL_NED' or v.get('system')!=1 or v.get('component')!=1:continue
  if v.get('mask',0)&448:continue # acceleration must actually be active
  row=np.asarray([0.,0.,0.]+v['acceleration_enu'],dtype=np.float32)
  if np.isfinite(row).all():echoes.append((float(v['stamp']),k,row))
 echoes.sort(key=lambda v:v[0]);used=set();confirmed=[]
 previous_time=float(rx_t[first[last][0]]) if last>=0 else -np.inf
 for i in raw_missing:
  if i<=last:continue # a mid-mission gap is not a truncated tail
  key=tuple(wire[i].tolist())
  if counts[key]!=1:continue # repeated identical commands are ambiguous
  for stamp,k,row in echoes:
   if k in used or stamp<=previous_time:continue
   if not sent[i]['state_stamp']-.004<=stamp<=sent[i]['state_stamp']+.08:continue
   if stamp>summary['mission_end_sim']+1e-9:continue
   if not np.array_equal(row,wire[i]):continue
   j=len(rx_t);rx_t=np.r_[rx_t,stamp];payload=np.vstack([payload,np.asarray(row,dtype=float)])
   error=float(np.max(abs(np.asarray(sent[i]['velocity_enu']+sent[i]['acceleration_enu'])-payload[j])))
   matched.append((i,j,error));first[i]=(j,error);used.add(k);previous_time=stamp;last=i
   confirmed.append(dict(controller_index=i,confirmation_stamp=stamp,source='PX4 POSITION_TARGET_LOCAL_NED -> vehicle_local_position_setpoint',receive_trace_index=k,unique_wire_payload=True,wire_payload=row.tolist(),maximum_payload_error=error))
   break
 return matched,first,rx_t,payload,dict(ULog_only_unmatched=raw_missing,PX4_downstream_confirmed=confirmed,timestamp_scope='ULog input publication or PX4 downstream setpoint publication; downstream time is an upper bound on reception, not isolated network latency')
