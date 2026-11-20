"""Reference results/7 layout with quantities actually used by the new method."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from scipy.spatial import ConvexHull
from PIL import Image
from render_geometry import Replay,axis3d,save
from render_suite import read,scope
from result_paths import result_root
ROOT=Path(__file__).resolve().parent

def draw(mode):
    folder=result_root()/mode/'breakout10/detour';a=read(folder);a['_fast_render']=True
    h=a['history'];b=a['barriers'];fig=plt.figure(figsize=(13.2,7.4))
    ax=axis3d(fig,[-.015,.15,.73,.70]);header=fig.text(.035,.877,'',fontsize=8,color='#335d72')
    fig.text(.035,.95,'十障碍三维动态突破',fontsize=17,color='#173c50')
    fig.text(.035,.915,'RA-DPCBF + 可执行绕行  |  '+scope(a),fontsize=10,color='#557381')
    rep=Replay(fig,ax,a,header)
    barax=fig.add_axes([.76,.60,.20,.23]);lineax=fig.add_axes([.76,.35,.20,.17]);psi_ax=fig.add_axes([.76,.12,.20,.17])
    bars=barax.bar(np.arange(10),np.zeros(10),color='#d5e8ef',label='参考增量')
    actual=barax.bar(np.arange(10),np.zeros(10),width=.36,color='#368b9b',label='实际增量')
    barax.set(ylim=(0,1),xlim=(-.7,9.7),xticks=np.arange(10),xticklabels=np.arange(1,11),ylabel='半径增量 / ΔR',title='10 个障碍的自适应半径')
    barax.legend(fontsize=7,ncol=2,loc='upper right');barax.tick_params(labelsize=7)
    for aa,values,label in [(lineax,b[:,:,6].min(axis=1),'最小动态净空 (m)'),(psi_ax,np.nanmin(b[:,:,12],axis=1),'最小 Ψ (m/s²)')]:
        aa.plot(h[:,0],values,c='#2b8394',lw=1);aa.axhline(0,c='.5',lw=.7,ls='--');aa.set(xlabel='时间 (s)',ylabel=label)
        aa.tick_params(labelsize=7);aa.spines[['top','right']].set_visible(False);aa.grid(alpha=.15)
    cursor1=lineax.axvline(0,c='#b27836',lw=1);cursor2=psi_ax.axvline(0,c='#b27836',lw=1)
    hull=ConvexHull(a['obstacles'][0,:,:2]);eq=hull.equations;status=fig.text(.035,.10,'',fontsize=9,color='#335d72')
    fig.text(.035,.047,'半径柱：浅色为 Rref−R0，深色为 R−R0，均除以 ΔR；它们不是碰撞概率。全部 10 个障碍进入硬约束。',fontsize=8,color='#557381')
    fig.text(.035,.018,'暖色为完整三维速度曲面：按参考风格平移显示（p_o + 0.62 s × w），不是实体外壳；洋红为绕行方向。',fontsize=8,color='#557381')
    def update(k):
        rep.update(k);base=.45+a['obstacles'][k,:,9];span=a['parameters'][5]
        for i in range(10):bars[i].set_height((b[k,i,5]-base[i])/span);actual[i].set_height((b[k,i,3]-base[i])/span)
        cursor1.set_xdata([h[k,0],h[k,0]]);cursor2.set_xdata([h[k,0],h[k,0]])
        escaped=bool(np.any(eq[:,:2]@h[k,1:3]+eq[:,2]>0));d=a['detour'][k]
        state=f'绕行对象 {int(d[1])+1}，χ={d[5]:.2f} m/s，ν={d[8]:.2f} m/s²' if d[0]>0 else '绕行约束未启用'
        status.set_text(('已离开初始障碍中心凸包' if escaped else '仍在初始障碍中心凸包内')+'  |  '+state)
        return []
    dest=folder/'figures';update(a['event']);save(fig,dest/'breakout_dashboard',dpi=180)
    ids=np.rint(np.linspace(0,len(h)-1,36)).astype(int);animation=FuncAnimation(fig,update,frames=ids,interval=100)
    animation.save(dest/'breakout_dashboard.gif',writer=PillowWriter(fps=10),dpi=80);plt.close(fig)
    with Image.open(dest/'breakout_dashboard.gif') as im:frames=im.n_frames;im.seek(frames-1);im.load()
    metadata=dict(frames=frames,source_times=h[ids,0].tolist(),normalized_bar_formula=['(Rref-R0)/delta_R','(R-R0)/delta_R'],
      probability_model=False,source_history_sha256=hashlib.sha256((folder/'history.npz').read_bytes()).hexdigest(),
      complete_3d_surfaces=True,max_boundary_equation_residual=rep.max_error)
    (dest/'breakout_dashboard_audit.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
    print('Rendered breakout dashboard: '+mode,flush=True)

if __name__=='__main__':
    for mode in ['reference','bounded']:draw(mode)
