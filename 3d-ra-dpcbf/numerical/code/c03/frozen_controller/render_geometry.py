"""Complete three-dimensional paraboloid surfaces from the saved h_i=0.

Reference-style display: p_world = p_obstacle + 0.62*w_boundary. This velocity
overlay is not a physical obstacle shell. Rendering never affects the QP.
"""
from pathlib import Path
import argparse,json,hashlib,gc
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
from matplotlib.patches import Circle
from PIL import Image
from ra_dpcbf.core import los_basis

ROOT=Path(__file__).resolve().parent
SCALE=.62
RHO_LIMIT=3.0
VELOCITY_Q_WINDOW=(-12.,3.)
NAMES={'default8':'交错运动 · 8 个障碍物','surrounded10':'动态包围 · 10 个障碍物','dense50':'密集动态环境 · 50 个障碍物',
       'breakout10':'动态突破 · 10 个障碍物','main10':'多方向交会 · 10 个障碍物',
       'head_on_static':'零横向初速度 · 静态迎面障碍','head_on_dynamic':'零横向初速度 · 动态迎面障碍',
       'diagonal':'斜向接近与受限加速度','diagonal_ceiling':'斜向接近 · 上方障碍'}
NAMES.update({f'profile{i}':f'多航点三维穿行 · {n} 个障碍物' for i,n in enumerate([4,7,10,14,18],1)})
plt.rcParams.update({'font.family':'Microsoft YaHei','font.size':10,'axes.unicode_minus':False,'pdf.fonttype':42,
                     'figure.facecolor':'#ffffff','savefig.facecolor':'#ffffff','axes.linewidth':.6})
TEAL='#287f95';BLUE='#006d9d';ORANGE='#e39723'
CMAP=plt.get_cmap('viridis')
CAMERA=np.array([np.cos(np.deg2rad(28))*np.cos(np.deg2rad(-72)),np.cos(np.deg2rad(28))*np.sin(np.deg2rad(-72)),np.sin(np.deg2rad(28))])

class WideAxes(Axes3D):
    def apply_aspect(self,position=None):
        self.set_position(self.get_position(original=True) if position is None else position,which='active')
    def get_proj(self):
        p=self.get_position(original=True);width,height=self.figure.get_size_inches()
        correction=np.eye(4);correction[0,0]=1/(p.width*width/(p.height*height))
        return correction@super().get_proj()

def axis3d(fig,rect):
    ax=WideAxes(fig,rect);fig.add_axes(ax);return ax

def faces(points):
    return np.stack([points[:-1,:-1],points[1:,:-1],points[1:,1:],points[:-1,1:]],axis=2).reshape(-1,4,3)

theta,phi=np.meshgrid(np.linspace(0,2*np.pi,21),np.linspace(0,np.pi,13))
SPHERE=np.stack([np.sin(phi)*np.cos(theta),np.sin(phi)*np.sin(theta),np.cos(phi)],axis=2)
SPHERE_FACES=faces(SPHERE)
NORMALS=SPHERE_FACES.mean(axis=1);NORMALS/=np.maximum(np.linalg.norm(NORMALS,axis=1)[:,None],1e-12)
LIGHT=np.array([-.4,-.6,1.]);LIGHT/=np.linalg.norm(LIGHT)
SHADING=.64+.36*np.maximum(0,NORMALS@LIGHT)
tt,pp=np.meshgrid(np.linspace(0,2*np.pi,13),np.linspace(0,np.pi,7))
LOW_FACES=faces(np.stack([np.sin(pp)*np.cos(tt),np.sin(pp)*np.sin(tt),np.cos(pp)],axis=2))
LOW_NORMALS=LOW_FACES.mean(axis=1);LOW_NORMALS/=np.maximum(np.linalg.norm(LOW_NORMALS,axis=1)[:,None],1e-12)
LOW_SHADING=.64+.36*np.maximum(0,LOW_NORMALS@LIGHT)

def basis(a,k,i):
    r=a['obstacles'][k,i,:3]-a['history'][k,1:4];e=r/np.linalg.norm(r)
    b1=np.cross(np.array([0.,0.,1.]),e)
    if np.linalg.norm(b1)<1e-8:b1,b2=los_basis(e)
    else:b1/=np.linalg.norm(b1);b2=np.cross(e,b1)
    return e,b1,b2

def boundary_geometry(a,k,i):
    b=a['barriers'][k,i];par=a['parameters'];beta=par[1]/(par[0]*par[19])
    e,b1,b2=basis(a,k,i);coefficient=beta*b[6];vertex=b[4]-b[6]/par[0]
    fine=not a.get('_fast_render',False)
    rho,theta=np.meshgrid(np.linspace(0,RHO_LIMIT,13 if fine else 9),np.linspace(0,2*np.pi,37 if fine else 25))
    z1=rho*np.cos(theta);z2=rho*np.sin(theta)
    q=vertex-coefficient*rho*rho
    w=q[:,:,None]*e+z1[:,:,None]*b1+z2[:,:,None]*b2
    # Display origin follows the corresponding obstacle, as in reference results/7.
    # This translation is only a glyph layout; the velocity gallery uses true q,z.
    origin=a['obstacles'][k,i,:3]
    points=origin+SCALE*w
    normals=e+2*coefficient*(z1[:,:,None]*b1+z2[:,:,None]*b2)
    residual=q-b[4]+b[6]/par[0]+beta*b[6]*(z1*z1+z2*z2)
    return points,q,normals,origin+SCALE*vertex*e,float(np.max(np.abs(residual)))

class Replay:
    def __init__(self,fig,ax,a,header=None,labels=True,boundary_function=None):
        self.fig,self.ax,self.a,self.header=fig,ax,a,header;self.n=a['obstacles'].shape[1]
        self.boundary_function=boundary_function or boundary_geometry
        self.palette=plt.get_cmap('YlOrRd')(np.linspace(.40,.85,self.n))
        self.labels=labels
        self.balls=Poly3DCollection([],linewidths=0,edgecolors='none',zsort='average')
        self.shells=Poly3DCollection([],linewidths=0,edgecolors='none',zsort='average')
        self.surfaces=Poly3DCollection([],linewidths=0,edgecolors='none',zsort='average')
        for collection in (self.balls,self.shells,self.surfaces):ax.add_collection3d(collection)
        self.vertices=ax.scatter([],[],[],s=16,c=ORANGE,depthshade=False)
        self.rims=Line3DCollection([],linewidths=.9);ax.add_collection3d(self.rims)
        self.obstacle_tails=Line3DCollection([],linewidths=.9);ax.add_collection3d(self.obstacle_tails)
        self.velocity_arrows=Line3DCollection([],linewidths=1.05);ax.add_collection3d(self.velocity_arrows)
        self.relative_arrows=Line3DCollection([],linewidths=1.25);ax.add_collection3d(self.relative_arrows)
        self.detour_arrow=Line3DCollection([],colors='#a44784',linewidths=2.5);ax.add_collection3d(self.detour_arrow)
        self.robot=Line3DCollection([],colors='#173c50',linewidths=2.9);ax.add_collection3d(self.robot)
        self.rotors=ax.scatter([],[],[],s=22,c='white',edgecolors=BLUE,linewidths=1,depthshade=False)
        self.traj=Line3DCollection([],cmap='Blues',norm=colors.Normalize(0,2.2),linewidths=3)
        ax.add_collection3d(self.traj)
        self.ids=[]
        if labels and self.n<=10:
            self.ids=[ax.text(0,0,0,str(i+1),color=TEAL,fontsize=8) for i in range(self.n)]
        ax.scatter(*a['initial'][:3],s=25,color='#219c7c',edgecolors='white',depthshade=False)
        ax.scatter(*a['goal'],s=105,marker='*',color=ORANGE,edgecolors='white',depthshade=False)
        ax.text(*a['goal']+[0,0,.45],'目标',fontsize=9,color='#986626')
        path=np.r_[a['initial'][None,:3],a.get('waypoints',a['goal'][None,:])]
        ax.plot(*path.T,color='#93aab7',lw=.7,ls='--',alpha=.6)
        w=a['workspace'];lo=w[:,0];hi=w[:,1]
        # Reference-sized arena, fixed camera, fixed limits for the full replay.
        ax.set(xlim=(lo[0]-4.3,hi[0]+.8),ylim=(lo[1]-.6,hi[1]+.6),zlim=(lo[2]-.5,hi[2]+.7),
               xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)')
        ax.set_box_aspect(hi-lo+np.array([5.1,1.2,1.2]),zoom=1.39);ax.set_proj_type('ortho');ax.view_init(elev=28,azim=-72)
        ax.tick_params(labelsize=8,pad=0);ax.set_facecolor('white');ax.grid(True)
        for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
            axis.pane.set_facecolor((.96,.98,.99,1));axis.pane.set_edgecolor('#d3e2eb')
            axis._axinfo['grid'].update(color='#dce8ee',linewidth=.45)
        self.max_error=0.

    def arrow_segments(self,p,d):
        length=np.linalg.norm(d)
        if length<1e-10:return []
        e=d/length;v=np.cross(e,np.array([0.,0.,1.]))
        if np.linalg.norm(v)<1e-8:v=np.array([1.,0.,0.])
        v/=np.linalg.norm(v);tip=p+d;head=min(.15,.20*length)
        return [np.array([p,tip]),np.array([tip,tip-head*e+.35*head*v]),np.array([tip,tip-head*e-.35*head*v])]

    def update(self,k):
        a=self.a;h=a['history'];o=a['obstacles'][k];b=a['barriers'][k];p=h[k,1:4]
        balls=[];ballcols=[];shells=[];shellcols=[];surfs=[];surfcols=[];lines=[];linecols=[]
        tails=[];arrows=[];relative=[];relativecols=[];vertices=[]
        dt=a['metrics']['control_dt'];tail=max(0,k-int(1.2/dt))
        # Visibility affects presentation only: all N rows remain in the QP.
        ring=a['metrics']['scene'] in ('surrounded10','breakout10')
        chosen=set(range(self.n)) if self.n<=10 and not ring else set(np.argsort(b[:,0])[:3].tolist())
        for i,row in enumerate(o):
            near=b[i,0]<6.
            sphere,shade=(SPHERE_FACES,SHADING) if not a.get('_fast_render',False) and (self.n<=10 or near) else (LOW_FACES,LOW_SHADING)
            balls.extend(row[:3]+row[9]*sphere)
            col=np.tile(colors.to_rgba(TEAL),(len(sphere),1));col[:,:3]*=shade[:,None]
            col[:,3]=.94 if self.n<=10 else .78;ballcols.extend(col)
            if near or self.n<=10:
                shell=LOW_FACES if a.get('_fast_render',False) else SPHERE_FACES
                shells.extend(row[:3]+b[i,3]*shell)
                shellcols.extend(np.tile([.48,.77,.94,.075],(len(shell),1)))
            if self.ids:
                self.ids[i].set_position((row[0],row[1]));self.ids[i].set_3d_properties(row[2]+row[9]+.13,zdir='z')
            surface,qmesh,normals,vertex,error=self.boundary_function(a,k,i);self.max_error=max(self.max_error,error)
            # A fixed velocity window. Full 360-degree two-dimensional surfaces,
            # with no parabolic section lines and no curvature-based rescaling.
            sf=faces(surface);qfaces=faces(np.stack([qmesh,qmesh,qmesh],axis=2))[:,:,0]
            visible=np.all((qfaces>=VELOCITY_Q_WINDOW[0])&(qfaces<=VELOCITY_Q_WINDOW[1]),axis=1)
            if self.n>10 and b[i,0]>=5.:visible[:]=False
            if i not in chosen:visible[:]=False
            if np.any(visible):
                surfs.extend(sf[visible]);normal=faces(normals).mean(axis=1)[visible]
                normal/=np.linalg.norm(normal,axis=1)[:,None]
                normal=np.where((normal@CAMERA<0)[:,None],-normal,normal)
                light=np.array([-.35,-.45,.82]);light/=np.linalg.norm(light)
                brightness=.75+.25*np.maximum(0,normal@light)
                rgba=np.tile(self.palette[i],(len(brightness),1));rgba[:,:3]*=brightness[:,None]
                rgba[:,3]=.50 if i==a['focus'] else .30
                surfcols.extend(rgba);vertices.append(vertex)
                rim=surface[:,-1].copy()
                rim[(qmesh[:,-1]<VELOCITY_Q_WINDOW[0])|(qmesh[:,-1]>VELOCITY_Q_WINDOW[1])]=np.nan
                lines.append(rim);linecols.append(self.palette[i])
            track=a['obstacles'][tail:k+1:10,i,:3]
            if len(track)>1:tails.append(track)
            arrows.extend(self.arrow_segments(row[:3],.75*row[3:6]))
            if near and i==a['focus']:
                parts=self.arrow_segments(p,SCALE*(row[3:6]-h[k,4:7]));relative.extend(parts)
                relativecols.extend([self.palette[i]]*len(parts))
        for artist,verts,col in [(self.balls,balls,ballcols),(self.shells,shells,shellcols),(self.surfaces,surfs,surfcols)]:
            artist.set_verts(np.asarray(verts).reshape(-1,4,3));artist.set_facecolor(col if len(col) else [(0,0,0,0)])
        self.vertices._offsets3d=np.array(vertices).T if vertices else ([],[],[])
        self.rims.set_segments(lines);self.rims.set_color(linecols)
        self.obstacle_tails.set_segments(tails);self.obstacle_tails.set_color((.16,.5,.59,.35))
        self.velocity_arrows.set_segments(arrows);self.velocity_arrows.set_color((.13,.37,.44,.8))
        self.relative_arrows.set_segments(relative);self.relative_arrows.set_color(relativecols if relativecols else [(0,0,0,0)])
        det=a.get('detour')
        self.detour_arrow.set_segments(self.arrow_segments(p,.9*det[k,2:5]) if det is not None and det[k,0]>0 else [])
        xyz=h[:k+1:3,1:4]
        self.traj.set_segments(np.stack([xyz[:-1],xyz[1:]],axis=1) if len(xyz)>1 else [])
        self.traj.set_array(np.linalg.norm(h[:k:3,4:7],axis=1))
        yaw=np.arctan2(h[k,5],h[k,4]);c,s=np.cos(yaw),np.sin(yaw);ends=[];rotors=[]
        for normal in [np.array([c,s,0.]),np.array([-s,c,0.])]:
            arm=p+np.array([-1.,1.])[:,None]*.27*normal;ends.append(arm);rotors.extend(arm)
        self.robot.set_segments(ends);rotors=np.array(rotors);self.rotors._offsets3d=rotors.T
        if self.header is not None:
            gap=float(np.min(b[:,0]-.3-o[:,9]))
            self.header.set_text(f't = {h[k,0]:5.2f} s     速度 {np.linalg.norm(h[k,4:7]):.2f} m/s     实体间隙 {gap:.3f} m     {self.n} 个障碍全部进入 QP')
        return []

def save(fig,path,dpi=200):
    fig.savefig(path.with_suffix('.png'),dpi=dpi)
    fig.savefig(path.with_suffix('.pdf'),dpi=dpi)

