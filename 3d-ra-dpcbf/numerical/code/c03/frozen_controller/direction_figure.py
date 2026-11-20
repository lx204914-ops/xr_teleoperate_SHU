"""Direction capacity from the actual full safe polytope, not an invented cone."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_suite import read
from render_geometry import save
from detour import polytope,support
from result_paths import result_root
ROOT=Path(__file__).resolve().parent

def plot():
    a=read(result_root()/'bounded/diagonal/detour');det=a['detour'];k=int(np.flatnonzero(det[:,11]==1)[0]);j=int(det[k,1])
    x=a['history'][k,1:7];o=a['obstacles'][k];b=a['barriers'][k];r=o[j,:3]-x[:3];D=np.linalg.norm(r);e=r/D
    w=o[j,3:6]-x[3:];q=e@w;z=w-q*e;kp=a['metrics']['config']['detour']['k_perp']
    n1=np.array([0.,0.,1.]);n1-=e*(e@n1);n1/=np.linalg.norm(n1);n2=np.cross(e,n1)
    theta=np.linspace(0,2*np.pi,181);A,c=polytope(b[:,9:12],b[:,8]-a['sampling_margins'][k]);gamma=[]
    for t in theta:
        d=np.cos(t)*n1+np.sin(t)*n2;sup,u=support(A,c,d);gamma.append(sup-d@o[j,6:9]-(kp-q/D)*(d@z))
    angle=np.arctan2(det[k,2:5]@n2,det[k,2:5]@n1)%(2*np.pi)
    fig,ax=plt.subplots(figsize=(7.4,3.8),layout='constrained');ax.plot(np.rad2deg(theta),gamma,c='#007c90',lw=1.8,label='共同安全输入集合中的方向能力 Γ')
    ax.scatter(np.rad2deg(angle),det[k,6],s=45,c='#a44784',label='实际启动方向')
    ax.axhline(det[k,8],c='#d18d24',ls='--',label='实际启动强度 ν');ax.axhline(0,c='.5',lw=.7)
    ax.set(xlim=(0,360),xlabel='法平面方向角 (°)，0° 为垂向',ylabel='Γ、ν (m/s²)',title=f'斜向迎面接近：全障碍约束下的实际方向能力，t={a["history"][k,0]:.3f} s')
    ax.spines[['top','right']].set_visible(False);ax.grid(alpha=.18);ax.legend(fontsize=9)
    dest=result_root()/'ablations';dest.mkdir(exist_ok=True);save(fig,dest/'direction_capacity',dpi=200);plt.close(fig)
if __name__=='__main__':plot()
