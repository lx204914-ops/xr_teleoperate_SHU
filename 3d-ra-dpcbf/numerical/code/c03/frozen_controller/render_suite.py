"""Data-derived replays, diagnostic figures and exact velocity-space surfaces."""
from pathlib import Path
import argparse,json,hashlib,gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image
from render_geometry import Replay,axis3d,save,boundary_geometry,NAMES,SCALE,RHO_LIMIT
from detour import polytope,support
ROOT=Path(__file__).resolve().parent

def read(folder):
    a=dict(np.load(folder/'history.npz'));a['metrics']=json.loads((folder/'metrics.json').read_text(encoding='utf8'))
    active=a['detour'][:,0]>0
    # Event is actual maximum reference-control correction, excluding terminal NaN.
    score=a['history'][:,15].copy();score[~np.isfinite(score)]=-np.inf
    a['event']=int(np.argmax(score));a['focus']=int(np.argmin(a['barriers'][a['event'],:,7]));return a

def scope(a):
    return '原始参考场景 · 保留离散反射/接触更新' if int(a['motion_mode'])==2 else '连续有界运动 · 参考初始布局'

def canvas(a):
    name=a['metrics']['scene'];fig=plt.figure(figsize=(12.8,7.2))
    ax=axis3d(fig,[.005,.13,.78,.72]);header=fig.text(.04,.88,'',fontsize=9,color='#335d72')
    fig.text(.04,.95,NAMES[name],fontsize=17,color='#173c50')
    fig.text(.04,.917,'RA-DPCBF + 可执行绕行    |    '+scope(a),fontsize=10,color='#557381')
    rep=Replay(fig,ax,a,header);info=fig.add_axes([.80,.24,.17,.56]);info.axis('off')
    note=info.text(0,1,'',va='top',fontsize=10,color='#335363',linespacing=1.9)
    cax=fig.add_axes([.21,.125,.40,.014]);fig.colorbar(ScalarMappable(norm=Normalize(0,2.2),cmap='Blues'),cax=cax,orientation='horizontal',label='机器人速度 (m/s)')
    cax.tick_params(labelsize=8,pad=1)
    fig.text(.04,.047,'青色实体与淡蓝安全球位于物理空间；暖色曲面是速度边界的显示图层，洋红箭头为当前绕行方向。',fontsize=8,color='#557381')
    fig.text(.04,.020,'按参考风格排布速度图层：p_o + 0.62 s × w；它不是实体表面。固定速度窗口；密集场景显示附近曲面，QP 保留全部障碍。',fontsize=7.4,color='#6d8590')
    def update(k):
        rep.update(k);h=a['history'][k];d=a['detour'][k];b=a['barriers'][k]
        on=d[0]>0;j=int(d[1]);finite=b[np.isfinite(b[:,12]),12];psi=float(finite.min()) if len(finite) else None
        res='目标到达' if k==len(a['history'])-1 and a['metrics']['reached_goal'] else '控制执行中'
        if k==len(a['history'])-1 and not a['metrics']['reached_goal']:res=a['metrics']['termination']
        psitext=f'{psi:.3f}' if psi is not None else '未获得可行输入'
        text=f"{res}\n\n全部 {len(b)} 个障碍\n硬安全约束\n\n最小 h：{b[:,7].min():.3f}\n最小 Ψ：{psitext}\n"
        if on:text+=f"\n绕行对象：{j+1}\nχ：{d[5]:.2f} m/s\nΓ可用：{d[7]:.2f} m/s²\nν：{d[8]:.2f} m/s²"
        else:text+='\n绕行约束：未启用'
        note.set_text(text);return []
    return fig,rep,update

def replay(folder,frames=48,gif=True):
    folder=folder.resolve()
    if tuple(folder.parts[-3:])==('reference','profile1','detour'):
        from render_profile1 import render,render_without_gif
        if gif:render(folder/'figures',frames=frames)
        else:render_without_gif(folder/'figures')
        return
    a=read(folder);fig,rep,update=canvas(a);figdir=folder/'figures';figdir.mkdir(exist_ok=True)
    update(len(a['history'])-1);save(fig,figdir/'trajectory_summary',dpi=180)
    update(a['event']);fig.savefig(figdir/'animation_preview.png',dpi=125)
    ids=np.unique(np.rint(np.linspace(0,len(a['history'])-1,frames)).astype(int))
    if gif:
        a['_fast_render']=True
        animation=FuncAnimation(fig,update,frames=ids,interval=80,blit=False)
        temp=figdir/'trajectory.pending.gif';animation.save(temp,writer=PillowWriter(fps=12),dpi=85)
        with Image.open(temp) as im:
            actual=im.n_frames;dimensions=list(im.size);im.seek(actual-1);im.load()
        if actual<len(ids)-1:raise RuntimeError('incomplete GIF')
        temp.replace(figdir/'trajectory.gif')
        a['_fast_render']=False
    meta=dict(source_history_sha256=hashlib.sha256((folder/'history.npz').read_bytes()).hexdigest(),
       source_mode=scope(a),rendered_source_indices=ids.tolist(),source_times=a['history'][ids,0].tolist(),
       actual_GIF_frames=actual if gif else 0,image_size=dimensions if gif else None,
       fixed_camera=[28,-72],full_angular_extent_rad=2*np.pi,surface_is_2D=True,parabolic_section_lines=False,
       source_equation='q=Rdot-s/Tr-beta*s*(z1*z1+z2*z2)',mapping='p_obstacle+.62*w_relative; display translation only',
       display_only_max_nearby_surfaces=(3 if a['obstacles'].shape[1]>10 or a['metrics']['scene'] in ('surrounded10','breakout10') else None),all_obstacles_constrained=True,velocity_rho_limit=3.,
       q_window=[-12.,3.],max_boundary_equation_residual=rep.max_error,raw_trajectory_no_fit=True)
    (figdir/'animation_audit.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8');plt.close(fig)
    diagnostics(folder,a);sequence(folder,a);gallery(folder,a);print('Rendered '+str(folder),flush=True)
    gc.collect()

def sequence(folder,a):
    t=a['history'][a['event'],0];end=a['history'][-1,0]
    times=[max(0,t-1.3),t,min(end,t+2.)];fig=plt.figure(figsize=(14.4,4.9))
    fig.text(.035,.95,NAMES[a['metrics']['scene']]+'：同次闭环的三个时刻',fontsize=14,color='#173c50')
    for j,t in enumerate(times):
        k=int(np.argmin(abs(a['history'][:,0]-t)));ax=axis3d(fig,[.005+j/3,.15,.33,.70])
        rep=Replay(fig,ax,a,labels=False);rep.update(k);ax.set_title(f't = {t:.2f} s',fontsize=10)
        ax.tick_params(labelsize=7)
    fig.text(.035,.045,scope(a)+'；全部曲面由对应时刻的状态生成，空间范围与速度映射倍率保持固定。',fontsize=9,color='#557381')
    save(fig,folder/'figures'/'avoidance_sequence',dpi=180);plt.close(fig)

def diagnostics(folder,a):
    h=a['history'];t=h[:,0];b=a['barriers'];d=a['detour'];fig,axes=plt.subplots(2,3,figsize=(13,6.3),layout='constrained')
    for axis in axes.ravel():
        axis.spines[['top','right']].set_visible(False);axis.grid(alpha=.16);axis.set_xlabel('时间 (s)')
    axes[0,0].plot(t,np.min(b[:,:,6],axis=1),label='最小动态净空 s',color='#19728d')
    body=b[:,:,0]-.3-a['obstacles'][:,:,9];axes[0,0].plot(t,body.min(axis=1),label='最小实体间隙',color='#d18c25')
    axes[0,0].axhline(0,c='.45',lw=.7);axes[0,0].set_ylabel('距离 (m)');axes[0,0].legend(fontsize=8)
    axes[0,1].plot(t,b[:,:,7].min(axis=1),label='最小 h',color='#19728d')
    axes[0,1].plot(t,np.nanmin(b[:,:,12],axis=1)/a['parameters'][3],label='最小 Ψ / k_h',color='#b76d35');axes[0,1].axhline(0,c='.45',lw=.7)
    axes[0,1].set_ylabel('h 与 Ψ/k_h (m/s)');axes[0,1].legend(fontsize=8)
    for i in range(b.shape[1]):axes[0,2].plot(t,b[:,i,3]-(.45+a['obstacles'][:,i,9]),lw=.8,alpha=.7)
    axes[0,2].set_ylabel(r'$R-R_0$ (m)');axes[0,2].set_title('各障碍的动态半径增量',fontsize=10)
    on=d[:,0]>0;chi=np.where(on,d[:,5],np.nan)
    axes[1,0].plot(t,chi,color='#a44784',label='实际 χ')
    kp=a['metrics']['config']['detour']['k_perp'];gmin=a['metrics']['config']['detour']['gamma_min']
    nmin=min(a['metrics']['config']['detour']['nu_req'],a['metrics']['config']['detour']['eta']*gmin)
    bound=np.full(len(t),np.nan)
    for segment in (np.unique(d[on,10]) if int(a['motion_mode'])==1 else []):
        ids=np.flatnonzero(on&(d[:,10]==segment));start=ids[0];delta=t[ids]-t[start]
        bound[ids]=np.exp(-kp*delta)*chi[start]+nmin/kp*(1-np.exp(-kp*delta))
    if int(a['motion_mode'])==1:axes[1,0].plot(t,bound,'--',c='#777777',lw=1,label='分段条件下界')
    else:axes[1,0].set_title('原始离散场景：不套用连续时间下界',fontsize=9)
    axes[1,0].set_ylabel('绕行相对速度 (m/s)');axes[1,0].legend(fontsize=8)
    axes[1,1].plot(t,np.where(on,d[:,7],np.nan),c='#19728d',label='Γ可用（扣采样余量）')
    axes[1,1].plot(t,np.where(on,d[:,8],np.nan),c='#d18c25',label='ν');axes[1,1].set_ylabel('加速度量 (m/s²)');axes[1,1].legend(fontsize=8)
    for i,label in enumerate([r'$u_x$',r'$u_y$',r'$u_z$']):axes[1,2].plot(t,h[:,7+i],lw=1,label=label)
    axes[1,2].axhline(2,c='.4',ls='--',lw=.8);axes[1,2].axhline(-2,c='.4',ls='--',lw=.8)
    axes[1,2].set_ylabel('控制加速度 (m/s²)');axes[1,2].legend(fontsize=8,ncol=3)
    fig.suptitle(NAMES[a['metrics']['scene']]+'：实际执行量与约束检查',fontsize=13)
    save(fig,folder/'figures'/'control_diagnostics',dpi=180);plt.close(fig)
    # Consistent metric axes in physical space, with true obstacle trails.
    fig,axs=plt.subplots(1,2,figsize=(10.5,3.7),layout='constrained')
    for ax,indices,title in zip(axs,[(0,1),(0,2)],['水平投影','垂直投影']):
        for i in range(a['obstacles'].shape[1]):ax.plot(a['obstacles'][::20,i,indices[0]],a['obstacles'][::20,i,indices[1]],lw=.65,alpha=.4,c='#3d8998')
        ax.plot(h[:,1+indices[0]],h[:,1+indices[1]],lw=2,c='#006d9d',label='实际轨迹')
        ax.scatter(*a['initial'][list(indices)],c='#219c7c',s=24);ax.scatter(*a['goal'][list(indices)],c='#d18c25',marker='*',s=85)
        ax.set(xlabel='X (m)',ylabel=('Y (m)' if indices[1]==1 else 'Z (m)'),title=title)
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    save(fig,folder/'figures'/'trajectory_projections',dpi=180);plt.close(fig)

def gallery(folder,a):
    # Every obstacle has its own surface, in actual LOS velocity coordinates.
    # Near-critical-time choice is mechanical: minimum h for that obstacle.
    n=a['obstacles'].shape[1];beta=a['parameters'][1]/(a['parameters'][0]*a['parameters'][19])
    ids=np.arange(n);critical=a['barriers'][a['barriers'][:,:,7].argmin(axis=0),ids]
    bottom=critical[:,4]-critical[:,6]/a['parameters'][0]-9*beta*critical[:,6]
    qmin=min(-16.,float(2*np.floor(bottom.min()/2)))
    for page,first in enumerate(range(0,n,10)):
        count=min(10,n-first);cols=min(5,count);rows=int(np.ceil(count/cols))
        fig=plt.figure(figsize=(3.05*cols,3.0*rows+.55))
        for pos,i in enumerate(range(first,first+count)):
            k=int(np.argmin(a['barriers'][:,i,7]));b=a['barriers'][k,i];s=b[6];lam=beta*s;vertex=b[4]-s/a['parameters'][0]
            ax=fig.add_subplot(rows,cols,pos+1,projection='3d');rr,tt=np.meshgrid(np.linspace(0,3,11),np.linspace(0,2*np.pi,33))
            X=rr*np.cos(tt);Y=rr*np.sin(tt);Q=vertex-lam*rr**2
            ax.plot_surface(X,Y,Q,color='#eaa741',edgecolor='none',alpha=.72,shade=True)
            ax.scatter(b[14],b[15],b[1],c='#007795' if b[7]>=0 else '#ad3b32',s=21,depthshade=False)
            ax.scatter(0,0,vertex,c='#555555',marker='x',s=14);ax.set(xlim=(-3,3),ylim=(-3,3),zlim=(qmin,3))
            ax.view_init(elev=24,azim=-58);ax.set_box_aspect([1,1,1.1]);ax.set_proj_type('ortho')
            ax.set_title(f'障碍 {i+1}   t={a["history"][k,0]:.2f}s\nλ={lam:.3f}, qv={vertex:.2f}',fontsize=8,pad=1)
            ax.set_xlabel('z1 (m/s)',fontsize=7,labelpad=-2);ax.set_ylabel('z2 (m/s)',fontsize=7,labelpad=-2);ax.set_zlabel('q (m/s)',fontsize=7,labelpad=-2)
            ax.tick_params(labelsize=6,pad=0)
        fig.suptitle('各障碍独立的三维速度边界；点为真实相对速度（蓝：h≥0，红：h<0），固定坐标范围',fontsize=10,y=.98)
        fig.subplots_adjust(left=.01,right=.99,bottom=.04,top=.85,hspace=.32,wspace=.05)
        fig.savefig(folder/'figures'/f'all_obstacle_surfaces_{page+1}.png',dpi=160);plt.close(fig)

def compare(output,folders):
    arrays=[read(f) for f in folders];labels={'no_detour':'关闭绕行','detour':'可执行绕行','fixed_direction':'固定方向绕行','fixed_radius':'固定半径 + 绕行'}
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    palette=['#c18430','#007795','#a44784','#4e9a66']
    for a,col in zip(arrays,palette):
        h=a['history'];name=labels.get(a['metrics']['variant'],a['metrics']['variant'])
        displacement=np.linalg.norm(h[:,1:4]-a['initial'][:3],axis=1)
        axes[0].plot(h[:,0],h[:,13],c=col,label=name)
        z=(a['obstacles'][:,0,3:6]-h[:,4:7]);r=a['obstacles'][:,0,:3]-h[:,1:4];e=r/np.linalg.norm(r,axis=1)[:,None]
        transverse=z-(z*e).sum(axis=1)[:,None]*e
        axes[1].plot(h[:,0],np.linalg.norm(transverse,axis=1),c=col,label=name)
        axes[2].plot(h[:,1],h[:,2],c=col,label=name)
    axes[0].set(xlabel='时间 (s)',ylabel='目标距离 (m)');axes[1].set(xlabel='时间 (s)',ylabel='法平面相对速度 (m/s)')
    axes[2].set(xlabel='X (m)',ylabel='Y (m)');axes[2].set_aspect('equal',adjustable='datalim')
    for ax in axes:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].legend(fontsize=8);fig.suptitle('同一迎面初态：安全保持与主动绕行分开验证',fontsize=13)
    output.mkdir(parents=True,exist_ok=True);save(fig,output/'detour_ablation',dpi=200);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folders',nargs='+');p.add_argument('--frames',type=int,default=48);p.add_argument('--no-gif',action='store_true')
    a=p.parse_args()
    for name in a.folders:replay(Path(name),a.frames,not a.no_gif)
