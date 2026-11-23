from pathlib import Path
import json,csv,html,shutil,hashlib
ROOT=Path(__file__).resolve().parents[1];DEST=Path('/mnt/c/Users/Public/Documents/CBF_Gazebo/65_first_arrival_comparison_20261102')
LABELS={'ra':'RA-DPCBF','ma_cbf_vo':'MA-CBF-VO','c3bf':'C3BF','park':'Park-inspired'}
def load(p):
 try:return json.loads(p.read_text())
 except (OSError,ValueError):return {}
def reason(summary):
 f=summary.get('failure') or ''
 if summary.get('completed'):return '到达'
 if 'initial_safety_condition' in f:return '初始条件未通过'
 if any(v in f for v in ('sampling_set_infeasible','deployment_set_infeasible','candidate_certificate_set_empty')):return '收紧可行集无解'
 if 'certificate' in f:return '采样认证未通过'
 if 'deadline' in f or 'control period' in f:return '控制周期超期'
 if 'horizon_expired' in f:return '观察期未到达'
 if 'Ground collision envelope' in f:return '地面安全包络触发'
 if 'Collision' in f or 'collision envelope' in f:return '实体安全包络触发'
 if 'qp_numerical' in f:return 'QP数值失败'
 return f or '未完成'
queue=load(ROOT/'queue_status.json');records=queue.get('completed',[]);rows=[];table=[]
for record in records:
 name=record['session'];p=ROOT/'runs'/name;from terminal_receipts import apply;apply(p);s=load(p/'summary.json');a=load(p/'independent_check.json');c=load(p/'continuous_check.json');f=load(p/'failed_cycle.json')
 row=dict(scene=record['case'],method=record['method'],run=name,outcome=reason(s),arrived=bool(s.get('completed')),execution_passed=bool(c.get('passed')),arrival_truth_s=a.get('first_arrival_truth_s'),path_m=(a.get('path_to_truth_arrival_m') if s.get('completed') else None),path_until_stop_m=a.get('path_to_truth_arrival_m'),minimum_body_clearance_m=a.get('minimum_body_clearance_m'),tracking_peak_mps2=c.get('tracking_peak_conservative_mps2',a.get('acceleration_tracking_peak_mps2')),complete_cycle_max_ms=(c.get('complete_planned_cycle_ms') or {}).get('max',f.get('finish_phase_ms')),processing_window_max_ms=(c.get('complete_active_cycle_ms') or {}).get('max'),actual_CBF_minimum=c.get('actual_sampled_CBF_min',a.get('minimum_actual_CBF_sampled')),failure=s.get('failure'),terminal_policy=s.get('terminal_policy'),nominal_cruise_mps=s.get('nominal_cruise_mps'),scene_sha256=s.get('scene_sha256'),no_landing_command=not any(v.get('command')==21 for v in s.get('acknowledgments',[])),observation_horizon_s=s.get('observation_horizon_s'))
 rows.append(row)
 media=ROOT/'native_recording'/name
 visual=''
 for filename,label in (('gazebo_native_live.gif','原生实录'),('native_terminal.png','原生终止画面')):
  if (media/filename).exists():visual=f'<a href="../native_recording/{name}/{filename}" target="_blank">{label}</a>';break
 if (media/'native_midframe.png').exists():visual+=f' · <a href="../native_recording/{name}/native_midframe.png" target="_blank">PNG</a>'
 def val(v,n=3):return '—' if v is None else f'{v:.{n}f}'
 issue=''
 if row['arrived'] and not row['execution_passed']:
  if (row['tracking_peak_mps2'] or 0)>.5:issue='跟踪超限'
  elif (row['processing_window_max_ms'] or 0)>20:issue='处理窗口超限'
  elif c.get('planned_cycles_over_20ms',0):issue='截止超期'
  elif c.get('unmatched_active_command_indices'):issue='指令匹配未通过'
  else:issue='执行检查未通过'
 row['execution_issue']=issue
 outcome=row['outcome']+('（'+issue+'）' if issue else '')
 table.append('<tr>'+''.join('<td>'+v+'</td>' for v in (str({'four':4,'ten':10,'fifty':50,'hundred':100}[row['scene']]),LABELS[row['method']],html.escape(outcome),val(row['arrival_truth_s']),val(row['path_m']),val(row['minimum_body_clearance_m']),val(row['tracking_peak_mps2']),val(row['complete_cycle_max_ms'],2),val(row['processing_window_max_ms'],2),visual))+'</tr>')
final=ROOT/'final';final.mkdir(exist_ok=True)
if rows:
 with (final/'核心对比数据.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
assert all(v['nominal_cruise_mps']==1.5 for v in rows if v['nominal_cruise_mps'] is not None)
assert all(v['no_landing_command'] for v in rows)
page=f'''<!doctype html><meta charset="utf-8"><title>PX4四方法首次到达对比</title><style>body{{max-width:1380px;margin:28px auto;padding:0 18px;font:16px/1.65 "Microsoft YaHei",sans-serif;color:#24333d}}h1{{font-size:26px}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:9px 7px;text-align:center;border-bottom:1px solid #e0e4e8}}thead{{border-top:2px solid;border-bottom:1px solid}}.note{{font-size:14px;color:#586774}}</style>
<h1>PX4＋Gazebo：四方法首次到达对比</h1><p>已完成{len(rows)}/16项。四、十、五十和一百障碍均复用已有场景，每场景四组共用运动、名义MPC、1.5 m/s巡航设置、20 ms控制周期及0.5 m/s²跟踪预算。首次物理真值进入0.25 m目标区域就停止计时、冻结虚拟场景并结束SITL，不做停稳观察或降落。</p>
<table><thead><tr><th>障碍数</th><th>方法</th><th>结果</th><th>到达/s</th><th>路径/m</th><th>最小间隙/m</th><th>跟踪峰值/(m/s²)</th><th>截止相位峰值/ms</th><th>处理窗口峰值/ms</th><th>实际画面</th></tr></thead><tbody>{''.join(table)}</tbody></table>
<p class="note">100障碍沿用已批准的40秒观察；其他场景保留既有时长。四障碍采用先前统一的接管初始化，其余场景统一悬停接管。到达与执行检查分别报告；QP无解、周期超期或超时均保留真实停止记录。未到达时，到达时间和到达路径留空；其他空白表示有效控制数据不足，不补造轨迹。截止相位从计划周期起点计；处理窗口从该次状态读取前计，二者起点不同，分别报告。仅为有限案例对比，不是成功率统计。</p>
<p class="note">画面直接来自原生Gazebo。MA只显示真实相对速度方向；C3BF使用标注的固定尺寸方向示意锥；Park使用其实际系数的抛物线截面；RA使用实际动态抛物面和风险半径。方向示意锥不冒充精确可行域。四组使用同一PX4本地执行分支、推力标定、姿态前馈及官方X500模型；不是实体机实测。</p><p><a href="核心对比数据.csv">完整CSV与运行来源</a> · <a href="../comparison_contract.json">公平配置</a></p>'''
(final/'四方法首次到达对比.html').write_text(page+'<p class="note">首入即终止时，仅对成功终止前最后4 ms内尚未确认的末包单列保留，不冒充已接收；途中指令匹配及其他预算检查保持不变。<a href="../qa/final_comparison_check.json">查看核对记录</a></p>',encoding='utf8')
DEST.mkdir(parents=True,exist_ok=True)
files=[]
for folder in ('final','code','frozen_controller','frozen_baselines','qa','scenes'):
 files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and not any(v in p.parts for v in ('__pycache__','cache')))
for name in ('comparison_contract.json','queue_status.json','source_lock.json','frozen_baseline.json','execution_contract.json','display_contract.json','model_geometry.json','common_initialization.json'):
 if (ROOT/name).exists():files.append(ROOT/name)
for record in records:
 for folder in ('runs','native_recording'):
  files.extend(p for p in (ROOT/folder/record['session']).rglob('*') if p.is_file() and p.name not in ('physical_audit_data.npz','continuous_audit_data.npz','continuous_sequence_audit_data.npz'))
hashes={}
for p in files:
 key=p.relative_to(ROOT).as_posix();d=DEST/key;h=hashlib.sha256(p.read_bytes()).hexdigest()
 if not d.exists() or hashlib.sha256(d.read_bytes()).hexdigest()!=h:d.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,d)
 hashes[key]=h
(DEST/'delivery_manifest.json').write_text(json.dumps(dict(scope='16 finite real comparisons; terminal first truth entry, no landing',files=hashes),indent=2))
print('REPORT_AND_NATIVE_RESULTS_UPDATED',len(rows),flush=True)
