from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Ellipse, Polygon, Circle
from matplotlib.collections import PolyCollection
from matplotlib.offsetbox import TextArea, HPacker, AnnotationBbox

OUT=Path(__file__).resolve().parent
for f in ['times.ttf','timesi.ttf','timesbd.ttf']:
    if Path('C:/Windows/Fonts/'+f).exists(): font_manager.fontManager.addfont('C:/Windows/Fonts/'+f)
plt.rcParams.update({'font.family':'Times New Roman','mathtext.fontset':'custom',
    'mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic',
    'mathtext.bf':'Times New Roman:bold','font.size':16})
TEAL='#28737C'; BLUE='#2A5275'; INK='#243D4B'; GRID='#DFE6EB'
RED='#C64B49'; PURPLE='#9272B4'
Tr,gamma,Zmax=2.,1.,4.
cases=[[(0.,-2.3),(0.,-.9)],[(5.,0.),(1.6,0.)],[(6.,.2),(3.,.3)]]
radii=[[2.2,2.2],[.75,1.7],[.65,1.25]]
titles=[r'(a) $\dot R_i$-only case ($s_i=0$)',
        r'(b) $s_i$-only case ($\dot R_i=0$)',
        r'(c) 3D-RADPCBF ($s_i,\dot R_i\ne 0$)']
# Orthographic camera; all six cells have identical projection and scale.
az,el=np.deg2rad(16),np.deg2rad(17)
cam=np.array([[np.cos(az),-np.sin(az),0],
              [-np.sin(az)*np.sin(el),-np.cos(az)*np.sin(el),np.cos(el)]])
depth=np.cross(cam[0],cam[1])
def project(points): return np.asarray(points)@cam.T
def line(ax, points, **kw):
    a=project(points); ax.plot(a[:,0],a[:,1],**kw)
def text3(ax, point, text, **kw):
    x,y=project(point); return ax.text(x,y,text,**kw)

def drone(ax, origin=(0,0,0),annotate=False):
    x,y=project(origin)
    arms=[(-.38,.18),(.38,.18),(-.38,-.18),(.38,-.18)]
    for dx,dy in arms:
        ax.plot([x,x+dx],[y,y+dy],color=BLUE,lw=2.8,zorder=15)
        ax.add_patch(Ellipse((x+dx,y+dy),.39,.145,angle=-8,fc='white',ec=BLUE,lw=1.7,zorder=17))
        ax.plot([x+dx-.10,x+dx+.10],[y+dy+.014,y+dy-.014],color=BLUE,lw=1.1,zorder=18)
        ax.add_patch(Circle((x+dx,y+dy),.025,fc=BLUE,zorder=19))
    ax.add_patch(Polygon([(x-.18,y+.12),(x+.19,y+.09),(x+.245,y-.067),(x-.17,y-.11)],
                         fc='#C9DFE8',ec=BLUE,lw=1.7,zorder=20))
    ax.scatter([x],[y],s=14,color=BLUE,zorder=23)
    if annotate:
        ax.text(x-.02,y+.42,'无人机',ha='center',va='bottom',color=BLUE,fontsize=18,zorder=22)
    ax.text(x+.10,y-.31,'0',ha='center',va='top',color=INK,fontsize=16,zorder=22)

def obstacle(ax,D,R,annotate=False):
    # Position geometry uses an independent common display scale, not velocity units.
    scale=.48
    x,y=project([D*scale,0,0]); r=R*scale
    # Translucent shaded sphere: preserve exact radius, remove wire meridians.
    tex=np.linspace(-1,1,240); xx,yy=np.meshgrid(tex,tex)
    rr2=xx*xx+yy*yy; zz=np.sqrt(np.maximum(0,1-rr2))
    illumination=np.clip(-.38*xx+.45*yy+.80*zz,0,1)
    rgba=np.zeros((*xx.shape,4))
    base=np.array([.69,.57,.81])
    rgba[:,:,:3]=base[None,None,:]*(.80+.20*illumination[:,:,None])
    rgba[:,:,3]=np.where(rr2<=1,.10+.25*zz,0)
    ax.imshow(rgba,extent=(x-r,x+r,y-r,y+r),origin='lower',interpolation='bilinear',zorder=12)
    ax.add_patch(Circle((x,y),r,fill=False,ec='#B9A5CE',lw=.65,alpha=.40,zorder=13))
    for j in range(32):
        rad=.22*(1-j/42)
        shade=.23+.20*j/31
        ax.add_patch(Circle((x-.0023*j,y+.0021*j),rad,fc=(shade,shade+.008,shade+.015),ec='none',zorder=14+j*.001))
    if annotate:
        ax.text(x+r+.10,y-.65,'障碍物',ha='left',va='center',fontsize=18,color='#515961',zorder=22)
    if annotate:
        angle=.85
        end=np.array([x+r*np.cos(angle),y+r*np.sin(angle)])
        ax.plot([x,end[0]],[y,end[1]],color=PURPLE,lw=1.05,zorder=16)
        ax.text(end[0]+.05,end[1]+.02,r'$R_i$',fontsize=20,color=PURPLE,zorder=20)
    line(ax,[(0,0,0),(D*scale,0,0)],color='#B9BFC3',ls=(0,(4,4)),lw=.9,zorder=1)
    # A distance bracket belongs to the physical scene and is separate from velocity axes.
    if annotate:
        by=-1.65
        ax.annotate('',xy=(x,by),xytext=(0,by),arrowprops={'arrowstyle':'|-|','lw':.8,'color':'#A0A9AF'},zorder=1)
        ax.text(x*.50,by-.08,r'$D_i$',ha='center',va='top',fontsize=20,color='#657A86')

def frame(ax,left_endpoint=-5.8):
    # No background grid: only the local velocity axes, anchored at the UAV.
    line(ax,[(left_endpoint,0,0),(0,0,0)],color='#BBC4C9',lw=.65,zorder=1)
    for end,label,offset in [((3.65,0,0),r'$q_i$',(.04,.13)),
                             ((0,6.3,0),r'$z_{i,1}$',(.14,-.30)),
                             ((0,0,2.4),r'$z_{i,2}$',(.05,.07))]:
        p=project(end)
        ax.annotate('',xy=p,xytext=(0,0),arrowprops={'arrowstyle':'->','color':'#8F9DA4','lw':1.0},zorder=10)
        ax.text(p[0]+offset[0],p[1]+offset[1],label,fontsize=16,color='#667780',
                bbox={'facecolor':'white','edgecolor':'none','pad':.6,'alpha':.90},zorder=25)

def old_boundary(ax,s,rdot):
    b=rdot-s/Tr; a=gamma*s/(Tr*Zmax)
    t=np.linspace(-1.86,1.86,160)
    line(ax,np.stack((b-a*t*t,np.zeros_like(t),t),axis=1),
         color='#91A4AD',lw=1.15,ls=(0,(4,3)),alpha=.8,zorder=2)
    line(ax,[(b,0,-2.05),(b,0,2.05)],color='#B5BFC5',lw=.8,ls=(0,(3,4)),zorder=2)

def surface(ax,s,rdot):
    b=rdot-s/Tr; a=gamma*s/(Tr*Zmax)
    if s==0:
        # The requested s_i=0 case is exactly a plane, NOT a paraboloid.
        corners=np.array([[b,-1.7,-1.86],[b,1.7,-1.86],
                          [b,1.7,1.86],[b,-1.7,1.86]])
        ax.add_collection(PolyCollection([project(corners)],facecolors='#C9DFEF',
            edgecolors='#689EBB',linewidths=1.15,alpha=.55,zorder=3))
        for y0 in np.linspace(-1.7,1.7,8):
            line(ax,[(b,y0,-1.86),(b,y0,1.86)],color='#81B4CF',alpha=.6,lw=.6,zorder=4)
        for z0 in np.linspace(-1.86,1.86,12):
            line(ax,[(b,-1.7,z0),(b,1.7,z0)],color='#81B4CF',alpha=.6,lw=.6,zorder=4)
        line(ax,[(b,0,-1.86),(b,0,1.86)],color=RED,lw=2.6,zorder=6)
        x,y=project([b,0,0]);ax.scatter([x],[y],s=25,c=RED,zorder=8)
        return {'s_i':s,'Rdot_i':rdot,'vertex':None,'q_intercept':b,
                'coefficient':0.,'max_residual':0.,'geometry':'plane'}
    theta=np.linspace(0,2*np.pi,49); rho=np.linspace(0,1.86,17)
    tt,rr=np.meshgrid(theta,rho)
    pts=np.stack((b-a*rr**2,rr*np.cos(tt),rr*np.sin(tt)),axis=-1)
    faces=[]; ds=[]
    for j in range(len(rho)-1):
        for k in range(len(theta)-1):
            face=pts[[j,j,j+1,j+1],[k,k+1,k+1,k]]
            faces.append(project(face));ds.append(np.mean(face@depth))
    order=np.argsort(ds)
    ax.add_collection(PolyCollection([faces[i] for i in order],facecolors='#C9DFEF',
                                     edgecolors='none',alpha=.40,zorder=3))
    for angle in np.linspace(0,2*np.pi,16,endpoint=False):
        line(ax,np.stack((b-a*rho**2,rho*np.cos(angle),rho*np.sin(angle)),axis=1),
             color='#81B4CF',alpha=.52,lw=.55,zorder=4)
    for radius in [.4,.8,1.2,1.55,1.86]:
        line(ax,np.stack((np.full_like(theta,b-a*radius**2),radius*np.cos(theta),radius*np.sin(theta)),axis=1),
             color='#689EBB' if radius==1.86 else '#81B4CF',alpha=.8 if radius==1.86 else .5,
             lw=1.25 if radius==1.86 else .55,zorder=5)
    t=np.linspace(-1.86,1.86,180)
    line(ax,np.stack((b-a*t*t,np.zeros_like(t),t),axis=1),color=RED,lw=2.6,zorder=6)
    x,y=project([b,0,0]);ax.scatter([x],[y],s=25,c=RED,zorder=8)
    return {'s_i':s,'Rdot_i':rdot,'vertex':b,'coefficient':a,
            'max_residual':float(np.abs(pts[:,:,0]-rdot+s/Tr+a*(pts[:,:,1]**2+pts[:,:,2]**2)).max())}


# Drawing glyphs/projection adapted from the project's 2026-11-17 figure source.
# Spatial glyphs and velocity-space surfaces use separate display scales.
import io,csv,hashlib
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]

def main():
 global Tr,gamma,Zmax
 config=ROOT/'numerical/code/conditional_6000/current_method_conditional/method_config.json'
 cfg=json.loads(config.read_text(encoding='utf8'))
 Tr=float(cfg['parameters'][0]);gamma=float(cfg['gamma_perp']);Zmax=float(cfg['Zmax'])
 plt.rcParams['font.family']='Microsoft YaHei'
 fig=plt.figure(figsize=(15,5.5),dpi=110,facecolor='white')
 axs=[fig.add_axes([.015+i*.33,.18,.32,.64]) for i in range(3)]
 fig.text(.5,.97,r'$h=0:\quad q_i=\dot R_i-s_i/T_r-\frac{\gamma_\perp s_i}{T_rZ_{\max}}(z_{i,1}^2+z_{i,2}^2)$',ha='center',va='top',fontsize=19)
 titles=['① 仅变化率：边界左右移动','② 仅距离余量：顶点与曲率变化','③ 两者共同变化']
 for i,title in enumerate(titles):fig.text(.175+i*.33,.84,title,ha='center',fontsize=14,color=INK)
 fig.text(.5,.085,'无人机、障碍物及紫色预留半径为空间示意；蓝色曲面为相对速度边界，两者比例独立。',ha='center',fontsize=11,color=INK)
 fig.text(.5,.035,'冻结状态的参数扫描，不是飞行轨迹；s=0时严格退化为平面，不是无人机外部的物理护盾。',ha='center',fontsize=10,color='#607383')
 frames=[];records=[];max_residual=0.
 for k in range(64):
  a=.5-.5*np.cos(2*np.pi*k/64)
  states=[(0.,-2.3+1.4*a,2.2),(1.8-1.25*a,0.,1.),(2.4-1.2*a,.2+.3*a,.65+.6*a)]
  for i,(ax,(s,rdot,R)) in enumerate(zip(axs,states)):
   ax.clear();ax.set_aspect('equal');ax.set_xlim(-5.9,4.3);ax.set_ylim(-2.35,2.65);ax.axis('off')
   frame(ax,-4.8);audit=surface(ax,s,rdot);drone(ax,annotate=True);obstacle(ax,s+R,R,annotate=True)
   max_residual=max(max_residual,float(audit['max_residual']))
   ax.text(.5,-.04,f'$s_i={s:.2f}$ m    '+r'$\dot R_i$'+f'$={rdot:+.2f}$ m/s',transform=ax.transAxes,ha='center',fontsize=13,color=TEAL)
   records.append({'帧':k,'板块':i+1,'距离余量s/m':s,'半径变化率/m·s⁻¹':rdot,'空间示意R/m':R,'空间示意D/m':s+R,'边界截点q/m·s⁻¹':rdot-s/Tr})
  buf=io.BytesIO();fig.savefig(buf,format='png',facecolor='white');buf.seek(0)
  with Image.open(buf) as im:im=im.convert('RGB');frames.append(im.copy())
  if k==16:frames[-1].save(ROOT/'numerical/figures/抛物面动态变化_预览.png')
 frames[0].save(ROOT/'numerical/media/抛物面动态变化.gif',save_all=True,append_images=frames[1:],duration=100,loop=0,optimize=False)
 plt.close(fig)
 with (ROOT/'numerical/data/抛物面动态变化_参数.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 report={'类型':'带空间示意图标的公式参数扫描，非实验轨迹','参数来源':str(config.relative_to(ROOT)),
  '参数文件SHA256':hashlib.sha256(config.read_bytes()).hexdigest(),'Tr':Tr,'beta':gamma/(Tr*Zmax),
  '最大h零边界残差':max_residual,'帧数':64,'空间与速度使用独立比例':True,'未执行控制器或生成飞行数据':True}
 (ROOT/'reproducibility/抛物面动画核验.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
