"""Truthful display of platform execution tests. No controller calls or trajectory edits."""
from pathlib import Path
import sys,json,hashlib,csv,argparse
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sys.path.insert(0,str(HERE));sys.dont_write_bytecode=True
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation,PillowWriter
from mpl_toolkits.mplot3d.art3d import Line3DCollection,Poly3DCollection
from scipy.spatial.transform import Rotation
from PIL import Image
import reference_style as style
OUT=HERE/"final"
METHODS=("ma_cbf_vo","c3bf","park","ra")
LABELS=("MA-CBF-VO","C3BF","Park-inspired","RA-DPCBF")
CFG=json.loads((ROOT/"frozen_controller/method_config.json").read_text("utf8"))
PAR=np.array(CFG["parameters"])
plt.rcParams.update({"font.family":"Times New Roman","font.size":12,"axes.labelsize":15,
    "axes.titlesize":18,"legend.fontsize":12,"axes.unicode_minus":False,"pdf.fonttype":42})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(method):
    folder=HERE/"results"/("four_obstacle_"+method+"_prioritized_execution")
    with np.load(folder/"history.npz",allow_pickle=False) as z:r={k:z[k] for k in z.files}
    r["summary"]=json.loads((folder/"summary.json").read_text("utf8"))
    r["source"]=str(folder/"history.npz");r["source_sha"]=sha(folder/"history.npz")
    with np.load(ROOT/"four_obstacle_scene.npz") as z:r.update(tape=z["tape"],td=float(z["tape_dt"]))
    return r
def drone(ax,p,q):
    R=Rotation.from_quat(q).as_matrix()
    seg=[];theta=np.linspace(0,2*np.pi,25)
    for sx,sy in ((-1,-1),(-1,1),(1,-1),(1,1)):
        c=np.array([sx*.35,sy*.35,0.])
        seg.append(np.stack([p,p+R@c]))
        ring=np.c_[.19*np.cos(theta),.16*np.sin(theta),np.zeros(len(theta))]+c
        seg.extend(np.stack([ring[:-1]@R.T+p,ring[1:]@R.T+p],axis=1))
    ax.add_collection3d(Line3DCollection(seg,colors="white",linewidths=3.,zorder=18))
    ax.add_collection3d(Line3DCollection(seg,colors="#46535C",linewidths=1.3,zorder=19))
    body=np.array([[-.21,-.11,.02],[.21,-.11,.02],[.26,0,.06],[.21,.11,.02],[-.21,.11,.02],[-.25,0,.06]])@R.T+p
    ax.add_collection3d(Poly3DCollection([body],facecolors="#C1CBD0",edgecolors="#34424B",linewidths=.7,zorder=19))
def basis(e):
    n=np.cross(e,[0.,0.,1.])
    if np.linalg.norm(n)<1e-9:n=np.cross(e,[0.,1.,0.])
    n/=np.linalg.norm(n)
    return n,np.cross(e,n)
def panel(ax,r,method,time,complete=False):
    # All panels share one fixed spatial frame. Failed panels stop at their
    # final recorded time; no simulated hovering is created.
    h=r["history"];t=min(float(time),float(h[-1,0]))
    k=max(0,min(len(h)-1,np.searchsorted(h[:,0],t+1e-9,side="right")-1))
    if method=="ra":k=min(k,len(r["barriers"])-1)
    t=float(h[k,0]);p=h[k,1:4];v=h[k,4:7];obs=style.state_at(r["tape"],t,r["td"])
    crop=np.array([[-2.,18.],[-6.,6.],[0.,8.]])
    style.style(ax,r,crop=crop);ax.set_box_aspect((20,12,8),zoom=1.25)
    ax.plot(*np.stack([r["initial"][:3],r["goal"]]).T,c="#AFB4B8",lw=.7,ls="--",zorder=1)
    arrival=r["summary"].get("first_arrival_s")
    if complete:
        end=len(h)
        ax.plot(*h[:end,1:4].T,c="#A73537",lw=.8,alpha=.25,zorder=9)
    index=np.unique(np.r_[np.arange(0,k+1,8),k]);points=h[index,1:4]
    if len(points)>1:
        trail=Line3DCollection(np.stack([points[:-1],points[1:]],axis=1),cmap=style.REDS,norm=style.NORM,lw=2.4,zorder=15)
        trail.set_array(np.linalg.norm(h[index[:-1],4:7],axis=1));ax.add_collection3d(trail)
    ax.scatter(*r["initial"][:3],s=22,c="#C8C8C8",depthshade=False,zorder=15)
    ax.scatter(*r["goal"],s=80,marker="*",c="#EDBD19",edgecolors="#977719",linewidths=.4,depthshade=False,zorder=16)
    segments=[];linecols=[];faces=[];facecols=[];geom_errors=[]
    if method=="ra":
        bi=min(k,len(r["barriers"])-1);bar=r["barriers"][bi]
        assert abs(h[bi,0]-t)<1e-8
        style.balls(ax,obs,bar[:,3],risk=True)
    for j,o in enumerate(obs):
        delta=o[:3]-p;D=np.linalg.norm(delta);e=delta/D;n1,n2=basis(e)
        w=o[3:6]-v;col=style.PALETTE[j%4];color=to_rgba(col,.82)
        if method in ("ma_cbf_vo","c3bf") and np.linalg.norm(w)>1e-8:
            # Direction arrows, no colored speed dots.
            a=.72*w/np.linalg.norm(w)
            ax.quiver(*p,*a,color=col,linewidth=1.,arrow_length_ratio=.25,zorder=12)
        if method=="c3bf":
            # Unit-direction cone: fixed max velocity radius; actual angular
            # aperture from this method's safety radius and current distance.
            radius=PAR[21]+PAR[22]+o[9];angle=np.arcsin(min(.999,radius/D))
            theta=np.linspace(0,2*np.pi,31);L=1.45
            ring=p+.62*L*(-np.cos(angle)*e+np.sin(angle)*(np.cos(theta[:,None])*n1+np.sin(theta[:,None])*n2))
            segments.append(ring);linecols.append(color)
            for a in range(0,30,6):segments.append(np.stack([p,ring[a]]));linecols.append(color)
            for a in range(30):faces.append([p,ring[a],ring[a+1]]);facecols.append(to_rgba(col,.045))
        elif method in ("park","ra"):
            if method=="park":
                radius=PAR[21]+PAR[22]+o[9];d=np.sqrt(max(D*D-radius*radius,1e-12))
                lam=.1*d/np.sqrt(w@w+1e-12);q0=-.5*d
                cap=min(style.RHO_CAP,np.sqrt(style.DEPTH/max(lam,1e-12)))
                rho=np.linspace(-cap,cap,61);q=q0-lam*rho*rho
                velocity=q[:,None]*e+rho[:,None]*n1
                x=o[:3]+style.SCALE*velocity
                segments.append(x);linecols.append(color)
                geom_errors.append(float(np.max(abs(q-q0+lam*rho*rho))))
            else:
                b=bar[j];lam=PAR[1]*b[6]/(PAR[0]*PAR[19]);q0=b[4]-b[6]/PAR[0]
                if lam<-1e-8:raise ValueError("Negative clearance: no fake paraboloid")
                cap=min(style.RHO_CAP,np.sqrt(style.DEPTH/max(lam,1e-12)))
                theta=np.linspace(0,2*np.pi,25);rho=np.linspace(0,cap,10)
                z1=np.cos(theta[:,None])*rho;z2=np.sin(theta[:,None])*rho
                q=q0-lam*(z1*z1+z2*z2)
                vel=q[:,:,None]*e+z1[:,:,None]*n1+z2[:,:,None]*n2
                x=o[:3]+style.SCALE*vel
                segments.append(x[:,-1]);linecols.append(color)
                for a in (0,6,12,18):segments.append(x[a]);linecols.append(color)
                for a in range(0,24,2):
                    for b in range(0,9,2):
                        faces.append([x[a,b],x[a+2,b],x[a+2,min(b+2,9)],x[a,min(b+2,9)]])
                        facecols.append(to_rgba(col,.025))
                geom_errors.append(float(np.max(abs(q-q0+lam*(z1*z1+z2*z2)))))
    if faces:ax.add_collection3d(Poly3DCollection(faces,facecolors=facecols,edgecolors="none",zorder=4))
    if segments:ax.add_collection3d(Line3DCollection(segments,colors=linecols,linewidths=.95,zorder=5))
    style.balls(ax,obs)
    ov=obs[:,3:6];speeds=np.linalg.norm(ov,axis=1)
    start=obs[:,:3]+obs[:,9,None]*ov/np.maximum(speeds[:,None],1e-12)
    ax.quiver(start[:,0],start[:,1],start[:,2],ov[:,0],ov[:,1],ov[:,2],length=.85,normalize=False,color="#DF950F",linewidth=1.,arrow_length_ratio=.25,zorder=11)
    drone(ax,p,h[k,19:23])
    return t,max(geom_errors,default=0.)
def figure(records,time,complete=False):
    fig=plt.figure(figsize=(14,9))
    checks=[]
    for j,(method,label) in enumerate(zip(METHODS,LABELS)):
        x=.018+.495*(j%2);y=.535 if j<2 else .075
        ax=fig.add_axes([x,y,.475,.425],projection="3d")
        t,error=panel(ax,records[method],method,time,complete)
        failed=records[method]["summary"]["mission"]!="goal"
        suffix="  [execution stop]" if failed and time>=records[method]["history"][-1,0]-1e-8 else ("  [record end]" if time>=records[method]["history"][-1,0]-1e-8 else "")
        fig.text(x+.012,.984 if j<2 else .515,
            f"({'abcd'[j]}) {label}  t = {t:.2f} s"+suffix,fontsize=16 if suffix else 18,va="top")
        checks.append(dict(method=method,time_s=t,geometry_residual=error))
    fig.legend([Line2D([],[],marker="o",c="#C8C8C8",ls="")],["Start"],loc="center",bbox_to_anchor=(.17,.039),frameon=False)
    fig.legend([Line2D([],[],marker="*",c="#EDBD19",ls="",markersize=10)],["Goal"],loc="center",bbox_to_anchor=(.50,.039),frameon=False)
    ca=fig.add_axes([.735,.033,.22,.012])
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=style.NORM,cmap=style.REDS),cax=ca,orientation="horizontal",ticks=[0,1.1,2.2])
    cb.ax.tick_params(labelsize=12,pad=1)
    fig.text(.714,.029,"v",fontstyle="italic",fontsize=15);fig.text(.97,.029,"[m/s]",fontsize=12)
    fig.text(.5,.003,"RotorPy six-DoF; local velocity-space overlays. Panels retain their final recorded frame when a recording ends.",ha="center",fontsize=10,color="#555555")
    return fig,checks
def render(gif=True):
    OUT.mkdir(exist_ok=True)
    records={name:load(name) for name in METHODS}
    fig,checks=figure(records,6.,complete=True)
    fig.savefig(OUT/"four_method_execution.png",dpi=400,facecolor="white")
    fig.savefig(OUT/"preview.png",dpi=100,facecolor="white");plt.close(fig)
    if gif:
        times=np.r_[np.arange(0,records["ra"]["history"][-1,0],.16),records["ra"]["history"][-1,0]]
        # One reusable canvas; drawing uses only recorded states and fixed tape.
        fig,_=figure(records,0.)
        def update(i):
            fig.clear()
            # Recreate on the same figure; avoid changing camera or limits.
            temp,_=figure(records,float(times[i]))
            temp.canvas.draw()
            im=np.asarray(temp.canvas.buffer_rgba()).copy()
            plt.close(temp)
            ax=fig.add_axes([0,0,1,1]);ax.axis("off");ax.imshow(im)
            return []
        # GIF frames are explicit Matplotlib renders, converted to indexed PNG
        # buffers by Pillow; no AI generation, no scene re-simulation.
        frames=[]
        for i,t in enumerate(times):
            f,_=figure(records,float(t));f.set_dpi(80);f.canvas.draw()
            frames.append(Image.fromarray(np.asarray(f.canvas.buffer_rgba()).copy()).convert("P",palette=Image.Palette.ADAPTIVE))
            plt.close(f)
            if i%25==0:print("GIF",i,"/",len(times),flush=True)
        frames[0].save(OUT/"four_method_execution_synchronized.gif",save_all=True,append_images=frames[1:],duration=160,loop=0,optimize=False)
        plt.close(fig)
    rows=[]
    for name in METHODS:
        s=records[name]["summary"]
        rows.append(dict(method=name,result=s["mission"],stop_or_arrival_s=s.get("first_arrival_s") or s.get("failure",{}).get("time_s") if s.get("failure") else s.get("first_arrival_s"),
            path_length_m=s.get("path_length_to_arrival_m"),minimum_clearance_m=s.get("min_body_gap_before_arrival_m"),
            acceleration_rmse_m_s2=s.get("acceleration_tracking_rmse_mps2"),
            minimum_actual_CBF_residual=s.get("min_actual_CBF_residual")))
    with (OUT/"four_method_execution_diagnostics.csv").open("w",encoding="utf-8-sig",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    manifest=dict(source={k:dict(path=r["source"],sha256=r["source_sha"]) for k,r in records.items()},
        deployment_protocol="200Hz, same common acceleration/attitude trim, shared priority tracking preference; no hard jerk constraint",
        postarrival_recording="Actual two-second settling check included. C3BF did not settle; no position or velocity was reset.",
        geometry_checks=checks,scene_sha256=sha(ROOT/"four_obstacle_scene.npz"),data_edits=False,
        overlay_mapping="X=obstacle_position+0.62*relative_velocity_boundary; local velocity schematic, not a physical shield",
        display_body="Enlarged quadrotor glyph; true quaternion. Physical radii unchanged.",
        cone="C3BF only; fixed maximum speed radius with actual aperture arcsin(R0/D). MA shows relative-velocity arrows.",
        comparison_scope="Shared 200Hz/common-trim/QP-reference protocol and safe-priority tracking preference; raw histories unchanged, including postarrival motion.",
        full_thin_trajectory_includes_two_second_settling_check=True)
    (OUT/"render_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf8")
    print("RENDER_DONE",flush=True)
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--no-gif",action="store_true");args=parser.parse_args()
    render(gif=not args.no_gif)
