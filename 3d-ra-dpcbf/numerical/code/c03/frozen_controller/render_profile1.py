"""Four-obstacle-only visual revision, replaying unmodified saved observations.

The world glyph uses RELATIVE velocity coordinates, translated to its vertex:
    x_display = p_uav + tau * (w_boundary - q_vertex * e).
This is not an obstacle shell or the unshifted relative-velocity boundary.
The separate velocity gallery retains the actual (z1,z2,q) boundary.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from render_geometry import Replay, axis3d, basis, faces, SPHERE_FACES, SHADING, save
from render_suite import read, diagnostics, gallery
from result_paths import result_root

ROOT = Path(__file__).resolve().parent
FOLDER = result_root() / 'reference/profile1/detour'
SCALE = .62  # seconds, fixed conversion from velocity to display length
RHO = 3.0   # m/s, fixed transverse velocity window
DELTA_Q = 4.0  # m/s, fixed vertex-relative axial window
PALETTE = ['#482475', '#31688e', '#35b779', '#d0cc20']
BG = '#ffffff'
INK = '#243b4a'


def risk_values(a, k):
    """Existing reference-radius command, not a newly invented risk model."""
    base = a['obstacles'][k, :, 9] + a['parameters'][21] + a['parameters'][22]
    return np.clip((a['barriers'][k, :, 5] - base) / a['parameters'][5], 0, 1)


def geometry(a, k, i):
    b = a['barriers'][k, i]
    beta = a['parameters'][1] / (a['parameters'][0] * a['parameters'][19])
    lam = beta * b[6]
    if lam < 0:
        raise ValueError('This display requires nonnegative clearance.')
    qv = b[4] - b[6] / a['parameters'][0]
    e, n1, n2 = basis(a, k, i)
    # A finite, full-azimuth patch. Curvature is never normalized or tuned.
    aperture = min(RHO, np.sqrt(DELTA_Q / lam)) if lam > 0 else RHO
    rr, tt = np.meshgrid(np.linspace(0, aperture, 13), np.linspace(0, 2*np.pi, 49))
    z1, z2 = rr*np.cos(tt), rr*np.sin(tt)
    q = qv-lam*rr**2
    centered_w = (q-qv)[..., None]*e + z1[..., None]*n1 + z2[..., None]*n2
    points = a['history'][k, 1:4] + SCALE*centered_w
    normals = e+2*lam*(z1[..., None]*n1+z2[..., None]*n2)
    residual = q-b[4]+b[6]/a['parameters'][0]+lam*(z1*z1+z2*z2)
    return points, q, normals, a['history'][k, 1:4].copy(), float(np.max(abs(residual)))


class PreviewReplay(Replay):
    def __init__(self, fig, ax, a, header=None, labels=True):
        super().__init__(fig, ax, a, header, labels, boundary_function=geometry)
        self.following = False
        self.palette = np.array([colors.to_rgba(c) for c in PALETTE])
        self.vertices.set_alpha(0)
        self.relative_arrows.set_visible(False)
        self.detour_arrow.set_visible(False)
        self.robot.set_color('#176ba0')
        self.robot.set_linewidth(3.2)
        # Velocity glyphs are a display overlay; keep the UAV apex legible.
        ax.computed_zorder = False
        for artist, order in [(self.traj,2),(self.surfaces,3),(self.rims,3.1),
                              (self.shells,4),(self.balls,5),(self.velocity_arrows,6),
                              (self.robot,10),(self.rotors,11)]:
            artist.set_zorder(order)
        for artist in ax.texts:
            if artist not in self.ids:artist.set_visible(False)
        for artist in ax.lines:artist.set_visible(False)
        for artist in ax.collections:
            if hasattr(artist,'_offsets3d') and artist not in (self.rotors,self.vertices):
                artist.set_alpha(0)
        self.traj.set_cmap('Blues')
        self.traj.set_linewidth(2.7)
        for artist in self.ids:
            artist.set_color('#222222')
            artist.set_fontsize(9)
            artist.set_ha('center')
            artist.set_bbox(dict(facecolor='white', edgecolor='none', alpha=.83, pad=1.5))
        self.shell_rings = Line3DCollection([], linewidths=.8)
        ax.add_collection3d(self.shell_rings)
        self.apex = ax.scatter([], [], [], c='#176ba0', s=24, depthshade=False)
        self.apex.set_zorder(12)
        ax.view_init(elev=40, azim=-70)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor((.98, .985, .99, 1))
            axis._axinfo['grid'].update(color='#e4e9ed', linewidth=.5)

    def update(self, k):
        a = self.a
        b = a['barriers'][k]
        o = a['obstacles'][k]
        p = a['history'][k, 1:4]
        near = int(np.argmin(b[:, 6]))
        a['focus'] = near
        super().update(k)
        risk = risk_values(a, k)
        ballverts, ballcols, shells, shellcols, rings, ringcols = [], [], [], [], [], []
        surfs, surfcols, rims, rimcols = [], [], [], []
        arrows, tails = [], []
        tt = np.linspace(0, 2*np.pi, 81)
        tail = max(0, k-int(1.4/a['metrics']['control_dt']))
        for i, row in enumerate(o):
            # Exact physical radius and actual generated R; no visual inflation.
            in_view = True  # All four obstacles, arrows and risk shells in every frame.
            col = np.ones((len(SPHERE_FACES), 4))
            col[:, :3] = (.035+.12*SHADING[:, None])
            if in_view:
                ballverts.extend(row[:3]+row[9]*SPHERE_FACES)
                ballcols.extend(col)
                shells.extend(row[:3]+b[i, 3]*SPHERE_FACES)
            proximity = np.clip(1-b[i, 6]/5., 0, 1)
            rgba = np.array(colors.to_rgba('#75bbdc'))
            rgba[3] = .08+.15*proximity
            if in_view:shellcols.extend(np.tile(rgba, (len(SPHERE_FACES), 1)))
            el,az=np.deg2rad([40.,-70.])
            right=np.array([-np.sin(az),np.cos(az),0.])
            up=np.array([-np.sin(el)*np.cos(az),-np.sin(el)*np.sin(az),np.cos(el)])
            for n1, n2 in [(right,up)]:
                if in_view:
                    rings.append(row[:3]+b[i, 3]*(np.cos(tt)[:, None]*n1+np.sin(tt)[:, None]*n2))
                    ringcols.append((.31, .63, .78, .25+.4*proximity))
            mesh, _, normals, apex, error = geometry(a, k, i)
            self.max_error = max(self.max_error, error)
            surfs.extend(faces(mesh))
            rgba = np.array(colors.to_rgba(PALETTE[i]))
            rgba[3] = .22
            light=np.array([-.4,-.6,.7]);light/=np.linalg.norm(light)
            normal=faces(normals).mean(axis=1);normal/=np.linalg.norm(normal,axis=1)[:,None]
            shading=.65+.35*abs(normal@light)
            facecolors=np.tile(rgba,(len(faces(mesh)),1));facecolors[:,:3]*=shading[:,None]
            surfcols.extend(facecolors)
            rims.append(mesh[:, -1]);rimcols.append((*colors.to_rgb(PALETTE[i]), .78))
            # Arrow starts at the surface of the black ball, not its occluded centre.
            vel = row[3:6]
            speed = np.linalg.norm(vel)
            if in_view and speed > 1e-10:
                arrows.extend(self.arrow_segments(row[:3]+row[9]*vel/speed, vel))
            if in_view:tails.append(a['obstacles'][tail:k+1:10, i, :3])
            if self.ids:
                text = self.ids[i]
                text.set_text(f'{i+1}  ρ={risk[i]:.2f}\nR={b[i,3]:.2f} m')
                text.set_position((row[0], row[1]))
                text.set_3d_properties(row[2]+b[i,3]+.3+.8*(i%2), zdir=None)
                text.set_visible(in_view)
                text.set_zorder(13)
        for artist, verts, rgba in [(self.balls, ballverts, ballcols), (self.shells, shells, shellcols),
                                    (self.surfaces, surfs, surfcols)]:
            artist.set_verts(np.asarray(verts).reshape(-1, 4, 3));artist.set_facecolor(rgba)
        self.rims.set_segments(rims);self.rims.set_color(rimcols);self.rims.set_linewidth(.9)
        self.shell_rings.set_segments(rings);self.shell_rings.set_color(ringcols)
        self.velocity_arrows.set_segments(arrows);self.velocity_arrows.set_color('#dc8b13');self.velocity_arrows.set_linewidth(1.9)
        self.obstacle_tails.set_segments(tails);self.obstacle_tails.set_color((.15, .15, .15, .22))
        self.apex._offsets3d = p[:, None]
        if self.header is not None:
            self.header.set_text(f't = {a["history"][k,0]:.2f} s     |     最近动态净空 {b[:,6].min():.2f} m     |     当前最近障碍 {near+1}')
        return []


def fixed_axes(ax,a,zoom=1.27):
    # One immutable world view for the whole recording. No obstacle culling.
    ax.set(xlim=(-1.2,33.2),ylim=(-1.,15.),zlim=(-1.,12.),
           xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
    ax.set_box_aspect([34.4,16.,13.],zoom=zoom)
    ax.set_proj_type('ortho');ax.view_init(elev=40,azim=-70)
    ax.set_title('',loc='left');ax.tick_params(labelsize=9,pad=1)
    ax.set_xticks(np.arange(0,33,5));ax.set_yticks(np.arange(0,15,3));ax.set_zticks(np.arange(0,13,3))


def canvas(a):
    fig=plt.figure(figsize=(14,8.4),facecolor=BG)
    fig.text(.045,.949,'3D RA-DPCBF · 四障碍动态避障',fontsize=17,color=INK)
    header=fig.text(.045,.913,'',fontsize=10,color='#4a5b65')
    ax=axis3d(fig,[.005,.115,.97,.76])
    rep=PreviewReplay(fig,ax,a,header)
    fixed_axes(ax,a)
    # Navigation context remains in the same physical-space pane.
    path=np.r_[a['initial'][None,:3],a['waypoints']]
    ax.plot(*path.T,color='#aeb5bc',lw=.8,ls='--',alpha=.7,zorder=1)
    ax.scatter(*a['initial'][:3],s=25,c='#307b65',depthshade=False,zorder=12)
    ax.scatter(*a['goal'],s=95,marker='*',c='#d95e43',depthshade=False,zorder=12)
    ax.text(*a['initial'][:3]+[0,0,-.65],'起点',fontsize=9,color='#307b65',zorder=14)
    ax.text(*a['goal']+[0,0,.65],'目标',fontsize=9,color='#b24731',zorder=14)
    handles=[Line2D([],[],marker='o',ls='',mfc='#111111',mec='#111111',label='动态障碍'),
             Line2D([],[],color='#dc8b13',lw=2,label='障碍速度'),
             Line2D([],[],marker='o',ls='',mfc='#c5e1ef',mec='#70acc9',label='实际风险球 R'),
             Line2D([],[],color='#176ba0',lw=2,label='无人机轨迹')]
    handles += [Line2D([],[],color=c,lw=2,label=f'抛物面 {i+1}') for i,c in enumerate(PALETTE)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.053),ncol=8,
               frameon=False,fontsize=9,handlelength=1.5,columnspacing=1.3)
    fig.text(.045,.025,'固定全场景视角，四个障碍始终同时显示。曲面为顶点对齐的速度边界图层；ρ 为归一化半径指令。',fontsize=8,color='#687c86')

    def update(k):
        rep.update(k)
        assert not rep.following and all(label.get_visible() for label in rep.ids)
        h=a['history'][k]
        header.set_text(f't = {h[0]:5.2f} s     |     无人机速度 {np.linalg.norm(h[4:7]):.2f} m/s     |     4 个动态障碍物')
        return []
    return fig,rep,update


def supporting_figures(folder,a):
    # Refresh the existing diagnostics, then style projections to match black obstacles.
    diagnostics(folder,a)
    gallery(folder,a)
    h=a['history'];b=a['barriers'];o=a['obstacles']
    fig,axs=plt.subplots(1,2,figsize=(11.5,4.3),layout='constrained')
    k=int(np.argmin(b[:,:,6].min(axis=1)))
    for ax,indices in zip(axs,[(0,1),(0,2)]):
        for i in range(4):
            ax.plot(o[::20,i,indices[0]],o[::20,i,indices[1]],c='#777777',lw=.8,alpha=.6)
            center=o[k,i,list(indices)]
            ax.add_patch(Circle(center,b[k,i,3],facecolor='#75bbdc',edgecolor='#68a9c5',lw=.8,alpha=.22))
            ax.add_patch(Circle(center,o[k,i,9],facecolor='#111111',edgecolor='none'))
            vel=o[k,i,3+np.array(indices)]
            ax.arrow(*(center+o[k,i,9]*vel/max(np.linalg.norm(vel),1e-12)),*(1.1*vel),width=.015,head_width=.17,color='#111111',length_includes_head=True)
            ax.text(*(center+[.2,b[k,i,3]+.13]),f'{i+1}  ρ={risk_values(a,k)[i]:.3f}',fontsize=8)
        ax.plot(h[:,1+indices[0]],h[:,1+indices[1]],c='#176ba0',lw=2,label='无人机实际轨迹')
        ax.scatter(*a['initial'][list(indices)],c='#368d79',s=30)
        ax.scatter(*a['goal'][list(indices)],marker='*',c='#dc9424',s=90)
        ax.set(xlabel='X (m)',ylabel=('Y (m)' if indices[1]==1 else 'Z (m)'))
        ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle(f'实际轨迹与动态安全球 · 球和箭头取 t={h[k,0]:.2f} s',fontsize=12)
    save(fig,folder/'figures/trajectory_projections',dpi=180);plt.close(fig)
    # Four distinct closest-approach episodes, not three copies of the same event.
    fig=plt.figure(figsize=(13.6,8))
    fig.text(.04,.96,'4 个障碍 · 各自最近接近时刻',fontsize=16,color=INK)
    for i in range(4):
        k=int(np.argmin(b[:,i,6]));ax=axis3d(fig,[.01+(i%2)*.5,.12+(1-i//2)*.40,.48,.32])
        rep=PreviewReplay(fig,ax,a,labels=True);rep.update(k)
        fixed_axes(ax,a,zoom=1.16)
        ax.set_title('',loc='left')
        ax.set_title(f'障碍 {i+1}  |  t={h[k,0]:.2f} s  |  s={b[k,i,6]:.2f} m',fontsize=10)
        ax.tick_params(labelsize=7)
    fig.text(.04,.035,'黑球及箭头来自实际障碍状态；蓝色包络采用实际 R。曲面均为完整旋转抛物面的有限面片，顶点与无人机对齐。',fontsize=9,color='#536b78')
    save(fig,folder/'figures/avoidance_sequence',dpi=170);plt.close(fig)


def supporting_to_output(output,a):
    temporary=output.parent/'supporting_render_20261017'
    temporary.mkdir(exist_ok=True)
    (temporary/'figures').mkdir(exist_ok=True)
    supporting_figures(temporary,a)
    for path in (temporary/'figures').iterdir():shutil.move(str(path),str(output/path.name))


def render_without_gif(output):
    """Refresh all stills without changing the existing movie or its audit."""
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True)
    a=read(FOLDER);fig,rep,update=canvas(a)
    update(a['event']);save(fig,output/'animation_preview',dpi=150)
    update(len(a['history'])-1);save(fig,output/'trajectory_summary',dpi=170)
    plt.close(fig);supporting_to_output(output,a)


def render(output,frames=100,static=False):
    a=read(FOLDER)
    if a['obstacles'].shape[1]!=4 or a['metrics']['scene']!='profile1':
        raise ValueError('This revision is restricted to the four-obstacle reference scene.')
    source_hash=hashlib.sha256((FOLDER/'history.npz').read_bytes()).hexdigest()
    output=output.resolve();output.mkdir(parents=True,exist_ok=True)
    fig,rep,update=canvas(a)
    near=int(np.argmin(a['barriers'][:,:,6].min(axis=1)))
    update(a['event']);save(fig,output/'animation_preview',dpi=150)
    if static:
        plt.close(fig);print('Static preview:',output,flush=True);return
    # Final frame shows the full flown path and all four current obstacle states.
    update(len(a['history'])-1);save(fig,output/'trajectory_summary',dpi=170)
    ids=np.unique(np.rint(np.linspace(0,len(a['history'])-1,frames)).astype(int))
    progress=[0]
    def animate(k):
        update(k);progress[0]+=1
        if progress[0]%20==0:print(f'Animation: {progress[0]}/{len(ids)} frames',flush=True)
        return []
    animation=FuncAnimation(fig,animate,frames=ids,interval=100,blit=False)
    temp=output/'trajectory.pending.gif';animation.save(temp,writer=PillowWriter(fps=10),dpi=100)
    with Image.open(temp) as im:
        count=im.n_frames;size=list(im.size)
        for f in range(count):im.seek(f);im.load()
    if count!=len(ids):raise RuntimeError('Missing GIF frames')
    temp.replace(output/'trajectory.gif');plt.close(fig)
    # Helpers write into folder/figures. Use only the selected output, leaving raw data untouched.
    supporting_to_output(output,a)
    audit=dict(source_history_sha256=source_hash,source_mode='original reference motion, unchanged',
        rendered_source_indices=ids.tolist(),source_times=a['history'][ids,0].tolist(),actual_GIF_frames=count,
        image_size=size,full_angular_extent_rad=2*np.pi,surface_is_2D=True,parabolic_section_lines=False,
        source_equation='q=qv-beta*s*(z1*z1+z2*z2); qv=Rdot-s/Tr',
        mapping='p_uav+.62*(w_boundary-qv*e); vertex-aligned RELATIVE-velocity display only',
        true_velocity_inset=False,display_apex_at_uav=True,velocity_rho_limit=RHO,
        vertex_relative_q_window=[-DELTA_Q,0.],max_boundary_equation_residual=rep.max_error,
        risk_definition='clip((Rref-R0)/(Rmax-R0),0,1); radius command, not collision probability',
        obstacle_sphere_radius='actual physical radius from saved observations',
        safety_sphere_radius='actual generated R, not desired/reference R',
        obstacle_arrow='actual vo; tail at physical surface, shaft=1.0 s * vo',
        camera='FIXED world view, elevation/azimuth 40/-70; x[-1.2,33.2], y[-1,15], z[-1,12]; no following or inset',
        all_4_obstacles_rendered_every_frame=True,no_spatial_culling=True,single_world_pane=True,
        reference_style='https://github.com/tkkim-robot/safe_control/tree/main/dynamic_env',
        all_obstacles_constrained=True,raw_trajectory_no_fit=True,revision='20261017_profile1_fixed_full_scene')
    (output/'animation_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    assert source_hash==hashlib.sha256((FOLDER/'history.npz').read_bytes()).hexdigest()
    print(json.dumps({'output':str(output),'frames':count,'source_hash':source_hash,'geometry_error':rep.max_error}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'preview_profile1_20261017/figures')
    parser.add_argument('--frames',type=int,default=100)
    parser.add_argument('--static',action='store_true')
    args=parser.parse_args();render(args.output,args.frames,args.static)
