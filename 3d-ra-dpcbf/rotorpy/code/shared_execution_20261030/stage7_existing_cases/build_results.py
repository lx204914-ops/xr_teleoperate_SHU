"""Render existing true RotorPy histories only; no controller reruns."""
from pathlib import Path
import sys,json,hashlib
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parent;ROOT=PACKAGE.parent
sys.path.insert(0,str(PACKAGE))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image
import reference_style as style
from render_results import drone
OUT=HERE/"final";OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family":"Times New Roman","font.size":12,"axes.labelsize":15,"axes.titlesize":18,"axes.unicode_minus":False})
receipt=json.loads((HERE/"validation.json").read_text("utf8"))
par=np.array(json.loads((ROOT/"frozen_controller/method_config.json").read_text("utf8"))["parameters"])
records=[];times=[];meta=[]
for entry,result in zip(receipt["sources"],receipt["results"]):
    folder=Path(entry["folder"]);source=folder/"history.npz"
    assert hashlib.sha256(source.read_bytes()).hexdigest()==entry["sha256"]
    with np.load(source,allow_pickle=False) as z:r={k:z[k] for k in z.files}
    h=r["history"];bar=r["barriers"];obs=r["obstacles"]
    assert len(h)==len(bar)+1
    phi=np.einsum("nij,nj->ni",bar[:,:,9:12],h[:-1,13:16])+bar[:,:,8]
    s=json.loads((folder/"summary.json").read_text("utf8"))
    assert abs(min(phi.min(),s["min_actual_CBF_residual"])-result["minimum_actual_CBF_including_endpoints"])<1e-8
    err=max(np.linalg.norm(h[:-1,13:16]-h[:-1,7:10],axis=1).max(),s["acceleration_tracking_max_norm_mps2"])
    assert abs(err-result["tracking_peak_including_endpoints"])<1e-8
    r["barrier_indices"]=np.arange(len(bar));r["parameters"]=par
    # Plot salience uses normalized raw commanded CBF residual, no KKT claim.
    r["sampling_margins"]=np.zeros(bar[:,:,12].shape)
    e=obs[:,:,:3]-h[:-1,None,1:4];e/=np.linalg.norm(e,axis=2)[:,:,None]
    r["basis_n1"]=style.frames(e)
    r["summary"]=s
    with np.load(ROOT/"tracked_interface/inputs"/f"smooth_N{entry['obstacles']:03d}_S2000.npz",allow_pickle=False) as z:
        r["tape"]=z["tape"];r["td"]=float(z["tape_dt"])
    gaps=np.min(np.linalg.norm(obs[:,:,:3]-h[:-1,None,1:4],axis=2)-par[21]-obs[:,:,9],axis=1)
    ids=np.flatnonzero(h[:-1,0]<=s["first_arrival_s"])
    k=int(ids[np.argmin(gaps[ids])]);times.append(float(h[k,0]))
    r["count"]=entry["obstacles"];records.append(r)
    meta.append(dict(obstacles=r["count"],snapshot_time_s=float(h[k,0]),history_sha256=entry["sha256"]))
def figure(ts,complete=False,dpi=100):
    fig=plt.figure(figsize=(14,5.5),dpi=dpi)
    for j,(r,t) in enumerate(zip(records,ts)):
        ax=fig.add_axes([.015+.5*j,.13,.475,.78],projection="3d")
        ii,k=style.snapshot(r,t);q=r["history"][k,19:23].copy()
        original=style.drone
        style.drone=lambda ax,p,heading,q=q:drone(ax,p,q)
        try:style.panel(ax,r,min(t,r["history"][-1,0]),complete=complete,crop=r["workspace"])
        finally:style.drone=original
        end=" [record end]" if t>=r["history"][-1,0]-1e-8 else ""
        fig.text(.028+.5*j,.975,f"({'ab'[j]}) {r['count']} obstacles   t = {r['history'][k,0]:.2f} s"+end,
                 fontsize=18 if not end else 15,va="top")
    fig.legend([Line2D([],[],marker="o",c="#C8C8C8",ls="")],["Start"],loc="center",bbox_to_anchor=(.17,.052),frameon=False)
    fig.legend([Line2D([],[],marker="*",c="#EDBD19",ls="",markersize=10)],["Goal"],loc="center",bbox_to_anchor=(.50,.052),frameon=False)
    ca=fig.add_axes([.745,.043,.205,.017])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=style.NORM,cmap=style.REDS),cax=ca,orientation="horizontal",ticks=[0,1.1,2.2]);cb.ax.tick_params(labelsize=12,pad=1)
    fig.text(.723,.038,"v",fontstyle="italic",fontsize=15);fig.text(.958,.038,"[m/s]",fontsize=12)
    fig.text(.5,.002,"RotorPy six-DoF; local velocity-space overlays, not physical shields.",ha="center",fontsize=10,color="#555555")
    return fig
f=figure(times,True);f.savefig(OUT/"ra_dense_physical.png",dpi=400,facecolor="white");f.savefig(OUT/"preview.png",dpi=100,facecolor="white");plt.close(f)
print("PNG_COMPLETE",times,flush=True)
frames=[]
end=max(r["history"][-1,0] for r in records)
for k,t in enumerate(np.r_[np.arange(0,end,.25),end]):
    f=figure([t,t],dpi=72);f.canvas.draw()
    frames.append(Image.fromarray(np.asarray(f.canvas.buffer_rgba()).copy()).convert("P",palette=Image.Palette.ADAPTIVE))
    plt.close(f)
    if k%20==0:print("GIF_FRAME",k,flush=True)
frames[0].save(OUT/"ra_dense_synchronized.gif",save_all=True,append_images=frames[1:],duration=250,loop=0,optimize=False)
tr=""
for row in receipt["results"]:
    tr+=f"<tr><th>{row['obstacles']}</th><td>{row['arrival_s']:.3f}</td><td>{row['minimum_body_clearance_m']:.3f}</td><td>{row['tracking_peak_including_endpoints']:.3f}</td><td>{row['minimum_actual_CBF_including_endpoints']:.3f}</td></tr>"
html='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>RotorPy复杂场景验证</title><style>
body{margin:0;background:#f3f5f6;color:#24313b;font:16px/1.8 "Microsoft YaHei",sans-serif}main{max-width:1200px;margin:28px auto;background:white;padding:30px 38px}h1{font-size:26px;margin:0}h2{font-size:19px;margin-bottom:6px}img{width:100%;height:auto}figure{margin:18px 0}figcaption,.note{font-size:14px;color:#59656e}table{border-collapse:collapse;width:100%;border-top:2px solid;border-bottom:2px solid}th,td{text-align:right;padding:10px}th:first-child{text-align:left}thead{border-bottom:1px solid}a{color:#145d6b}details{border:1px solid #dae1e5;padding:10px;margin-top:14px}summary{cursor:pointer}@media(max-width:650px){main{margin:0;padding:20px 12px}table{font-size:12px}td,th{padding:5px}}</style><main>
<h1>RotorPy复杂场景验证</h1>
<p>RA-DPCBF在已有十障碍、五十障碍平滑动态场景中均首次到达并停稳，无碰撞或安全QP失败。实测加速度跟踪误差均满足本轮0.5 m/s²预算。</p>
<p class="note">Crazyflie六自由度模型，真值状态、零风；外环5 ms，物理步长0.5 ms。场景与主方法保持不变。这是两个案例的物理执行验证，不是密度成功率统计。</p>
<figure><a href="ra_dense_physical.png"><img src="ra_dense_physical.png" alt="RA-DPCBF十障碍和五十障碍真实六自由度轨迹与彩色速度边界"></a><figcaption>分别选取保存状态中实体间隙最小的交会时刻；淡红细线为后续真实轨迹。灰球、橙色速度箭头与淡紫风险球沿用既有风格；每个障碍均有对应抛物面，较小原始屏障余量的边界着重显示。曲面是速度空间叠加示意，不是物理护盾。</figcaption></figure>
<details><summary>播放同步GIF</summary><img loading="lazy" src="ra_dense_synchronized.gif" alt="两个真实物理案例同步回放"><p class="note">记录结束后保持最后记录帧，未编造后续悬停或障碍运动。</p></details>
<h2>核心结果</h2><div style="overflow:auto"><table><thead><tr><th>障碍数</th><th>首次到达/s</th><th>最小实体间隙/m</th><th>跟踪误差峰值/(m/s²)</th><th>最小实际CBF残差</th></tr></thead><tbody>TABLE</tbody></table></div>
<p class="note">到达为首次进入目标0.25 m区域，之后继续真实飞行2 s；间隙统计至首次到达，误差和屏障复核覆盖完整记录及控制区间端点。</p>
<p>两组实际安全屏障和绕行条件的最小复核残差均为正，且未出现电机饱和。到达后2 s，两组位置误差均约3.2 mm，速度均小于0.007 m/s，完成了到达与停稳验证。</p>
<p>实时性仍未通过：完整控制计算P95分别为5.56和6.83 ms，超过5 ms周期。这些离线结果不能直接视为已具备200 Hz实时执行能力，也不能代替全工况误差保证或真机验证。</p>
<p class="note">初始化说明：十障碍采用四组共同可行姿态；五十障碍的C3BF原始初始输入集合无解，因此该项采用RA自身可行的悬停初始化，仅验证RA，不作为四方法对比。原始共同初始化停止记录保留。</p>
<p><a href="../physical_validation.csv">数据表</a> · <a href="../validation.json">复核记录</a> · <a href="../../final/四旋翼物理仿真结果.html">四方法物理对比</a></p></main></html>'''.replace("TABLE",tr)
(OUT/"复杂场景验证结果.html").write_text(html,encoding="utf8")
(OUT/"figure_manifest.json").write_text(json.dumps(dict(sources=meta,mapping="X=obstacle_position+0.62*relative_velocity_boundary",formulas_unchanged=True,all_obstacles_have_boundary=True,salience="four smallest normalized raw commanded CBF residuals; remaining boundaries retained",actual_quaternion=True),indent=2),encoding="utf8")
print("RENDER_DONE",flush=True)
