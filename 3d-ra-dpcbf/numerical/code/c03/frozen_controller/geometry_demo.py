"""Frozen-parameter geometry illustration, explicitly not a closed-loop result."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
from render_geometry import save
from result_paths import result_root
ROOT=Path(__file__).resolve().parent

def draw(dest,frames=48):
    dest.mkdir(parents=True,exist_ok=True);fig=plt.figure(figsize=(13,4.7))
    axes=[fig.add_subplot(1,3,k+1,projection='3d') for k in range(3)]
    artists=[None]*3
    rho,theta=np.meshgrid(np.linspace(0,3,20),np.linspace(0,2*np.pi,49))
    x=rho*np.cos(theta);y=rho*np.sin(theta)
    fig.suptitle('三维抛物面边界：顶点平移、开口改变及二者联动',fontsize=14,color='#173c50')
    fig.text(.04,.045,'几何参数对照，均取 R′=0：左列联动 Tr 与 γ 以固定 β；中列调整 γ；右列固定 Tr、γ，仅减小 s。',fontsize=8.5,color='#557381')
    fig.text(.04,.012,'各列均满足 λ=γs/(Tr Zmax)、qv=−s/Tr；Zmax=9，0≤γ≤1。固定速度坐标；这不是闭环仿真。',fontsize=8.5,color='#557381')
    def update(k):
        f=k/max(frames-1,1)
        # Each pair comes from admissible Tr,gamma,s and Rdot=0. The first two
        # are parameter comparisons; only the third represents clearance change
        # with the final controller's Tr=.8 and gamma=1 held fixed.
        pairs=[(2*.25/(.8*9),-2/(.8+2.4*f)),(2*(1-.9*f)/(.8*9),-2/.8),((2-1.6*f)/(.8*9),-(2-1.6*f)/.8)]
        for i,(ax,(lam,v)) in enumerate(zip(axes,pairs)):
            if artists[i] is not None:artists[i].remove()
            artists[i]=ax.plot_surface(x,y,v-lam*rho*rho,color=['#db9239','#c77548','#21889b'][i],alpha=.78,edgecolor='none')
            ax.set(xlim=(-3,3),ylim=(-3,3),zlim=(-5.5,1),xlabel='z1 (m/s)',ylabel='z2 (m/s)',zlabel='q (m/s)')
            ax.set_title(['固定 λ，顶点趋近零','固定顶点，曲面变平','顶点与形状共同变化'][i]+f'\nλ={lam:.2f}，qv={v:.2f}',fontsize=10)
            ax.view_init(elev=25,azim=-55);ax.set_box_aspect([1,1,1.1]);ax.set_proj_type('ortho');ax.tick_params(labelsize=7)
        return []
    fig.subplots_adjust(left=.015,right=.98,top=.80,bottom=.16,wspace=.12)
    update(frames-1);save(fig,dest/'paraboloid_parameter_effects',dpi=180)
    animation=FuncAnimation(fig,update,frames=range(frames),interval=100)
    animation.save(dest/'paraboloid_parameter_effects.gif',writer=PillowWriter(fps=10),dpi=85);plt.close(fig)

if __name__=='__main__':draw(result_root()/'geometry')
