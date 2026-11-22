"""Algebraic boundary sweep from frozen code parameters; NOT a flight rollout."""
from pathlib import Path
import json,hashlib,csv,io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]

def main():
 config=ROOT/'numerical/code/conditional_6000/current_method_conditional/method_config.json'
 cfg=json.loads(config.read_text(encoding='utf8'))
 tr=float(cfg['parameters'][0]);beta=float(cfg['gamma_perp'])/(tr*float(cfg['Zmax']))
 plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10})
 out=ROOT/'numerical/media';out.mkdir(exist_ok=True)
 angle=np.linspace(0,2*np.pi,36);rad=np.linspace(0,2,15)
 rr,aa=np.meshgrid(rad,angle);z1=rr*np.cos(aa);z2=rr*np.sin(aa)
 fig=plt.figure(figsize=(14,5),dpi=105)
 axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)]
 fig.text(.5,.035,'相对速度空间的公式参数扫描｜红点为边界顶点｜s=0时退化为平面｜示意动画，不是新增飞行实验',ha='center',fontsize=11,color='#43566A')
 frames=[];records=[];max_residual=0.
 for k in range(64):
  phase=2*np.pi*k/64;rd=.7*np.sin(phase);s=1.15+.85*np.cos(phase)
  cases=[(0.,rd),(s,0.),(s,rd)]
  for i,(ax,(gap,speed)) in enumerate(zip(axes,cases)):
   ax.clear();q=speed-gap/tr-beta*gap*(z1*z1+z2*z2)
   residual=q-speed+(1/tr+beta*(z1*z1+z2*z2))*gap
   max_residual=max(max_residual,float(np.max(np.abs(residual))))
   ax.plot_surface(q,z1,z2,color='#8DC7DC',alpha=.36,linewidth=0,shade=False)
   ax.plot_wireframe(q,z1,z2,rstride=5,cstride=3,color='#458CA3',linewidth=.55,alpha=.7)
   rimq=speed-gap/tr-beta*gap*4
   ax.plot(np.full_like(angle,rimq),2*np.cos(angle),2*np.sin(angle),color='#43879C',lw=1.2)
   vertex=speed-gap/tr;ax.scatter([vertex],[0],[0],color='#CB4F59',s=32,depthshade=False)
   ax.plot([-6.5,1.7],[0,0],[0,0],color='#8D9BA8',lw=.8)
   ax.set(xlim=(-6.5,1.7),ylim=(-2.2,2.2),zlim=(-2.2,2.2),xlabel=r'$q$ (m/s)',ylabel=r'$z_1$ (m/s)',zlabel=r'$z_2$ (m/s)')
   ax.set_box_aspect((1.6,1,1));ax.view_init(elev=19,azim=-65)
   ax.set_title(['① 仅半径变化率：边界左右平移','② 仅距离余量：位置与曲率变化','③ 两者共同变化'][i],fontsize=11,pad=10)
   ax.text2D(.02,.94,f's = {gap:.2f} m   '+r'$\dot R$'+f' = {speed:+.2f} m/s',transform=ax.transAxes,color='#275875')
   for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
   records.append({'帧':k,'板块':i+1,'距离余量s/m':gap,'半径变化率/m·s⁻¹':speed,'顶点q/m·s⁻¹':vertex,'曲率系数βs':beta*gap})
  fig.suptitle(r'$h=0:\quad q=\dot R-s/T_r-\beta s(z_1^2+z_2^2)$',fontsize=17,y=.98)
  fig.subplots_adjust(left=.02,right=.98,bottom=.12,top=.84,wspace=.05)
  buf=io.BytesIO();fig.savefig(buf,format='png',facecolor='white');buf.seek(0)
  frame=Image.open(buf).convert('RGB');frames.append(frame.copy())
  if k==16:frame.save(ROOT/'numerical/figures/抛物面动态变化_预览.png')
 frames[0].save(out/'抛物面动态变化.gif',save_all=True,append_images=frames[1:],duration=95,loop=0,optimize=False)
 plt.close(fig)
 with (ROOT/'numerical/data/抛物面动态变化_参数.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 (ROOT/'reproducibility/抛物面动画核验.json').write_text(json.dumps({'类型':'公式参数扫描，非实验轨迹','参数来源':str(config.relative_to(ROOT)),
  '参数文件SHA256':hashlib.sha256(config.read_bytes()).hexdigest(),'Tr':tr,'beta':beta,'最大h零边界残差':max_residual,'帧数':64,
  '范围':{'s':[.3,2.0],'Rdot':[-.7,.7]},'未执行控制器或生成飞行数据':True},ensure_ascii=False,indent=2),encoding='utf8')
 print('动画完成；最大边界残差：',max_residual)

if __name__=='__main__':main()
