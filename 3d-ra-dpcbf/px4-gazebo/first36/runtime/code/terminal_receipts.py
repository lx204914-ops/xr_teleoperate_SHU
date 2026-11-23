"""Distinguish a cancelled terminal packet from loss during an active interval."""
from pathlib import Path
import json
def apply(folder):
 p=Path(folder);check=p/'continuous_check.json'
 if not check.exists():return
 c=json.loads(check.read_text());s=json.loads((p/'summary.json').read_text())
 if s.get('terminal_policy')!='first_truth_entry_stop_SITL_no_landing':return
 trace=json.loads((p/'execution_trace.json').read_text());sent=[v for v in trace['send'] if 'controller_index' in v]
 raw=c.get('unmatched_command_indices',[]);cancelled=[]
 if raw and sent and s['completed']:
  i=len(sent)-1;gap=s['mission_end_sim']-sent[i]['world_sim_snapshot']
  if i in raw and -1e-8<=gap<=.004+1e-8:cancelled=[i]
 active=[i for i in raw if i not in cancelled]
 c.update(terminal_unconfirmed_packet_indices=cancelled,unmatched_active_command_indices=active,terminal_receipt_policy='Only the final packet sent within one 4ms physics step of successful termination may remain unconfirmed. Raw unmatched indices retained; all earlier packets must match.',terminal_receipt_scope='Simulator termination boundary, not evidence of an active-interval communication loss or of receipt of the terminal packet')
 if cancelled:
  backup=p/'continuous_check_before_terminal_classification.json'
  if not backup.exists():backup.write_bytes(check.read_bytes())
  c['passed']=bool(s['completed'] and s['failure'] is None and c['first_arrival_truth_s'] is not None and not active and not c['planned_cycles_over_20ms'] and c['tracking_peak_conservative_mps2']<=.5 and c['minimum_body_clearance_m']>0 and c['actual_sampled_CBF_min']>=-1e-6 and not c['motor_saturation_records'] and c['complete_active_cycle_ms']['max']<=20 and abs(c['actual_world_real_time_factor']-1)<.05 and c['final_goal_error_m']<.25)
 check.write_text(json.dumps(c,indent=2))
