"""Saved reference-style drawing helpers; no control or experiment imports."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.collections import PatchCollection
from matplotlib.patches import Ellipse
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
PALETTE=['#D5AE00','#7650AA','#2388CA','#279C70']
SCALE=.62
DEPTH=2.4
RHO_CAP=3.0
REDS=colors.ListedColormap(plt.colormaps['Reds'](np.linspace(.28,.98,256)))
NORM=colors.Normalize(0,2.2)


def frames(e):
    """Parallel transport only the display basis; never change LoS or curvature."""
    out=[];previous=None
    for row in e:
        arr=[]
        for j,v in enumerate(row):
            n=(np.cross(v,[0,0,1.]) if previous is None else previous[j]-v*np.dot(v,previous[j]))
            if np.linalg.norm(n)<1e-8:n=np.cross(v,[0,1.,0])
            n=n/np.linalg.norm(n);arr.append(n)
        previous=np.asarray(arr);out.append(previous.copy())
    return np.asarray(out)

def state_at(tape,t,td,mode=1):
    """Literal NumPy translation of the frozen scene_tapes.state_at(mode1)."""
    if mode!=1:raise ValueError('Only verified smooth mode1 is supported')
    if t>=(len(tape)-1)*td-1e-12:return tape[-1].copy()
    k=min(int(np.floor((t+1e-10)/td)),len(tape)-2);dt=np.clip(t-k*td,0,td)
    a=tape[k];b=tape[k+1];out=a.copy();jerk=(b[:,6:9]-a[:,6:9])/td
    out[:,:3]=a[:,:3]+dt*a[:,3:6]+.5*dt**2*a[:,6:9]+dt**3/6*jerk
    out[:,3:6]=a[:,3:6]+dt*a[:,6:9]+.5*dt**2*jerk
    out[:,6:9]=a[:,6:9]+dt*jerk
    return out

def boundary(r,i,j):
    b=r['barriers'][i,j];p=r['history'][r['barrier_indices'][i],1:4];o=r['obstacles'][i,j];e=(o[:3]-p);e=e/np.linalg.norm(e)
    n1=r['basis_n1'][i,j];n2=np.cross(e,n1);par=r['parameters'];lam=par[1]*b[6]/(par[0]*par[19]);q0=b[4]-b[6]/par[0]
    if lam < -1e-9:raise ValueError('Negative clearance coefficient: do not invent paraboloid')
    cap=min(RHO_CAP,np.sqrt(DEPTH/lam)) if lam>1e-12 else RHO_CAP
    th=np.linspace(0,2*np.pi,25);rho=np.linspace(0,cap,10)
    a=np.cos(th[:,None])*rho;b2=np.sin(th[:,None])*rho;q=q0-lam*(a*a+b2*b2)
    w=q[:,:,None]*e+a[:,:,None]*n1+b2[:,:,None]*n2
    x=o[:3]+SCALE*w
    return x,(e,n1,n2,lam,q0,cap,w)

class ScreenCollection(PatchCollection):
    def do_3d_projection(self):return 0

def balls(ax,obs,radii=None,risk=False):
    """Exact orthographic projection of physical sphere silhouettes, shaded in2D.

    The projected ellipse comes from the SVD of the3D->2D linear map, so display
    shading changes neither the supplied physical radius nor the data position.
    """
    M=ax.get_proj();p=obs[:,:3];x,y,z=proj3d.proj_transform(p[:,0],p[:,1],p[:,2],M)
    c=np.c_[x,y];jac=np.empty((len(p),2,3))
    for dim in range(3):
        pp=p.copy();pp[:,dim]+=1
        xx,yy,zz=proj3d.proj_transform(pp[:,0],pp[:,1],pp[:,2],M);jac[:,:,dim]=np.c_[xx-x,yy-y]
    patches=[];fc=[];ec=[]
    for j in np.argsort(z)[::-1]:
        U,s,_=np.linalg.svd(jac[j],full_matrices=False);rad=obs[j,9] if radii is None else radii[j]
        width,height=2*rad*s;angle=np.degrees(np.arctan2(U[1,0],U[0,0]))
        if risk:
            patches.append(Ellipse(c[j],width,height,angle=angle));fc.append((.68,.57,.82,.085));ec.append((.59,.43,.76,.18))
        else:
            for k,sc in enumerate(np.linspace(1,.12,15)):
                center=c[j]+np.array([-.12*width,.13*height])*(1-sc)
                patches.append(Ellipse(center,width*sc,height*sc,angle=angle))
                gray=.40+.27*k/14;fc.append((gray,gray,gray,1));ec.append((.32,.32,.32,.9) if k==0 else (0,0,0,0))
    coll=ScreenCollection(patches,facecolors=fc,edgecolors=ec,linewidths=.4 if not risk else .25,zorder=7 if not risk else 3)
    ax.add_collection(coll,autolim=False)

def drone(ax,p,heading):
    # Display-only recognizable quadrotor; physical robot radius stays unchanged.
    yaw=np.arctan2(heading[1],heading[0]);rot=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    seg=[];th=np.linspace(0,2*np.pi,29)
    for sx,sy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
        c=np.array([sx*.51,sy*.51,0]);seg.append(np.stack([p,p+rot@c]))
        ring=np.c_[.285*np.cos(th),.195*np.sin(th),np.zeros(len(th))]+c;seg.extend(np.stack([ring[:-1]@rot.T+p,ring[1:]@rot.T+p],axis=1))
    ax.add_collection3d(Line3DCollection(seg,colors='white',linewidths=3.0,zorder=18))
    ax.add_collection3d(Line3DCollection(seg,colors='#46535C',linewidths=1.3,zorder=19))
    body=1.5*np.array([[-.20,-.11,.02],[.20,-.11,.02],[.26,0,.06],[.20,.11,.02],[-.20,.11,.02],[-.25,0,.06]])@rot.T+p
    ax.add_collection3d(Poly3DCollection([body],facecolors='#C1CBD0',edgecolors='#34424B',linewidths=.75,zorder=19))

def style(ax,r,crop=None):
    ws=r['workspace'].copy() if crop is None else np.asarray(crop)
    ax.computed_zorder=False;ax.set_proj_type('ortho');ax.view_init(27,-64)
    extent=np.diff(ws,axis=1).ravel();aspect=extent.copy();aspect[1:]=np.maximum(aspect[1:],extent[0]*np.array([.50,.42]))
    ax.set_box_aspect(aspect,zoom=1.08 if crop is not None else 1.24)
    ax.set(xlim=ws[0],ylim=ws[1],zlim=ws[2],xlabel='X [m]',ylabel='Y [m]',zlabel='Z [m]')
    from matplotlib.ticker import MaxNLocator
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.set_major_locator(MaxNLocator(4));axis.pane.fill=False
        axis._axinfo['grid'].update(color=(.79,.82,.85,.55),linewidth=.5)
        axis.line.set_color('#87939B')
    ax.tick_params(labelsize=12,pad=-1);ax.xaxis.labelpad=3;ax.yaxis.labelpad=3;ax.zaxis.labelpad=1

def snapshot(r,t):
    h=r['history'];ids=r['barrier_indices'];ii=max(0,min(len(ids)-1,np.searchsorted(h[ids,0],min(t,h[-1,0])+1e-9,side='right')-1));k=int(ids[ii])
    return ii,k

def panel(ax,r,t,complete=False,crop=None,risktags=False):
    style(ax,r,crop);ii,k=snapshot(r,t);h=r['history'];p=h[k,1:4];b=r['barriers'][ii]
    terminated=t>h[-1,0]+1e-8
    obs=state_at(r['tape'],float(t),r['td']) if terminated else r['obstacles'][ii]
    ax.plot(*np.stack([r['initial'][:3],r['goal']]).T,c='#B0B5B9',lw=.65,ls='--',zorder=1)
    if complete:ax.plot(*h[:,1:4].T,c='#B8433E',alpha=.20,lw=1.,zorder=10)
    skip=np.unique(np.r_[np.arange(0,k+1,8),k]);pts=h[skip,1:4]
    if len(pts)>1:
        lc=Line3DCollection(np.stack([pts[:-1],pts[1:]],axis=1),cmap=REDS,norm=NORM,lw=2.5,zorder=15)
        lc.set_array(np.linalg.norm(h[skip[:-1],4:7],axis=1));ax.add_collection3d(lc)
    ax.scatter(*r['initial'][:3],s=18,c='#C8C8C8',depthshade=False,zorder=15)
    ax.scatter(*r['goal'],s=75,marker='*',c='#EDBD19',edgecolors='#977719',linewidths=.4,depthshade=False,zorder=16)
    line_segments=[];line_colors=[];line_widths=[];polys=[];face_colors=[]
    if not terminated:
        slack=(b[:,12]-r['sampling_margins'][ii])/np.maximum(np.linalg.norm(b[:,9:12],axis=1),1e-10)
        # Visual salience means small evaluated slack; not an uncomputed KKT dual.
        finite=np.flatnonzero(np.isfinite(slack))
        strong=set(finite[np.argsort(slack[finite])[:4]]);num=len(obs)
        for j in range(num):
            x,aux=boundary(r,ii,j);e,n1,n2,lam,q0,cap,w=aux;col=colors.to_rgb(PALETTE[j%4]);active=j in strong
            alpha=.92 if active else (.38 if num<=12 else .11)
            width=1.1 if active else (.65 if num<=12 else .32)
            curves=[x[:,-1,:]]
            curves.extend(x[a,:,:] for a in [0,6,12,18])
            for curve in curves:
                # A whole constant-colour polyline has exactly the same vertices
                # and edges, but avoids tens of thousands of3D projection calls.
                line_segments.append(curve);line_colors.append((*col,alpha));line_widths.append(width)
            # Subtle colored face under crisp meridians; plane remains a plane.
            if num<=12 or active:
                for aa in range(0,len(x)-1,2):
                    for bb in range(0,len(x[0])-1,2):
                        polys.append([x[aa,bb],x[min(aa+2,len(x)-1),bb],x[min(aa+2,len(x)-1),min(bb+2,len(x[0])-1)],x[aa,min(bb+2,len(x[0])-1)]])
                        face_colors.append((*col,.022 if num<=12 else .027))
        if polys:ax.add_collection3d(Poly3DCollection(polys,facecolors=face_colors,edgecolors='none',zorder=4,rasterized=True))
        ax.add_collection3d(Line3DCollection(line_segments,colors=line_colors,linewidths=line_widths,zorder=5))
        balls(ax,obs,b[:,3],risk=True)
    balls(ax,obs)
    # One common scale and true instantaneous velocities; arrows begin on sphere.
    v=obs[:,3:6];speed=np.linalg.norm(v,axis=1);start=obs[:,:3]+obs[:,9,None]*v/np.maximum(speed[:,None],1e-12)
    ax.quiver(start[:,0],start[:,1],start[:,2],v[:,0],v[:,1],v[:,2],length=.85,normalize=False,color='#DF950F',linewidth=.85,arrow_length_ratio=.25,zorder=11)
    if risktags and not terminated:
        par=r['parameters'];base=obs[:,9]+.3+.15
        expansion=np.clip((b[:,3]-base)/par[5],0,1)
        for o,val in zip(obs,expansion):ax.text(o[0],o[1],o[2]+o[9]+.22,f'{val:.2f}',fontsize=9,color='#5C536B',ha='center',zorder=16)
    drone(ax,p,h[k,4:7]);return ii,k
