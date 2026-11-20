"""Regenerate Chinese figures from archived records; no simulation imports."""
from pathlib import Path
import csv, json, collections, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parents[1]
METHODS=['ra_conditional_detour','c3bf','ma_cbf_vo','park_3d']
LABELS=['RA-DPCBF','C3BF','MA-CBF-VO','Park适配']
COLORS=['#253D94','#CE5078','#219D89','#E7A521']
COUNTS=[1,10,30,50,100]

def load(rel):
    with (ROOT/rel).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(rel,rows):
    p=ROOT/rel;p.parent.mkdir(exist_ok=True,parents=True)
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def save(fig,rel):
    p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(p.with_suffix('.png'),dpi=180,facecolor='white')
    fig.savefig(p.with_suffix('.svg'),facecolor='white')
    plt.close(fig)

def setup():
    available={f.name for f in font_manager.fontManager.ttflist}
    font=next((f for f in ['Microsoft YaHei','Noto Sans CJK SC','SimHei','SimSun'] if f in available),None)
    if not font:raise RuntimeError('请先安装中文字体：微软雅黑或 Noto Sans CJK SC。')
    plt.rcParams.update({'font.family':font,'font.size':11,'axes.unicode_minus':False,
      'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path',
      'axes.titlepad':12,'legend.frameon':False,'axes.labelpad':8})

def numerical():
    data=load('numerical/data/formal_6000.csv')
    keys={(r['scene_id'],r['method']) for r in data}
    assert len(data)==len(keys)==6000
    assert all(sum(r['method']==m for r in data)==1500 for m in METHODS)
    outcomes=['goal','timeout','qp_infeasible','numerical_failure','collision']
    assert set(r['outcome_class'] for r in data)<=set(outcomes)
    fig,axs=plt.subplots(1,4,figsize=(14,4.3),sharey=True,layout='constrained')
    ocolors=['#4779DB','#E5A52D','#B6A6DC','#CF5671','#454545']
    names=['到达目标','超时','QP不可行','数值失败','实体碰撞']
    for ax,m,label in zip(axs,METHODS,LABELS):
        bottom=np.zeros(5)
        for oc,c in zip(outcomes,ocolors):
            vals=np.array([sum(r['outcome_class']==oc for r in data if r['method']==m and int(r['obstacle_count'])==n)/3 for n in COUNTS])
            ax.bar(range(5),vals,bottom=bottom,color=c,width=.7)
            if oc=='goal':
                for i,v in enumerate(vals):
                    ax.text(i,v/2 if v>=5 else 2.5,f'{v:.1f}',ha='center',va='center',color='white' if v>=5 else 'black',fontsize=9)
            bottom+=vals
        ax.set(title=label,xticks=range(5),xticklabels=COUNTS,ylim=(0,105),xlabel='障碍物数量')
        ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    axs[0].set_ylabel('运行结果占比（%）')
    fig.legend([Patch(facecolor=c) for c in ocolors],names,loc='outside lower center',ncol=5)
    save(fig,'numerical/figures/6000次结果分类')
    aggregate=[]
    for m,label in zip(METHODS,LABELS):
        group=[r for r in data if r['method']==m]
        aggregate.append({'方法':label,'运行次数':len(group),'到达次数':sum(r['outcome_class']=='goal' for r in group),
          '完成率/%':round(sum(r['outcome_class']=='goal' for r in group)/len(group)*100,4),
          '超时次数':sum(r['outcome_class']=='timeout' for r in group),'QP不可行次数':sum(r['outcome_class']=='qp_infeasible' for r in group),
          '数值失败次数':sum(r['outcome_class']=='numerical_failure' for r in group),'实体碰撞次数':sum(r['outcome_class']=='collision' for r in group)})
    write('numerical/data/6000次总体汇总.csv',aggregate)
    cell=[]
    for m,label in zip(METHODS,LABELS):
        for n in COUNTS:
            for rad in [.3,.5,.7]:
                g=[r for r in data if r['method']==m and int(r['obstacle_count'])==n and abs(float(r['maximum_radius_m'])-rad)<1e-8]
                assert len(g)==100
                ok=[r for r in g if r['outcome_class']=='goal']
                cell.append({'方法':label,'障碍物数量':n,'最大障碍半径/m':rad,'运行次数':len(g),'到达次数':len(ok),
                  '完成率/%':len(ok),'成功样本到达时间中位数/s':float(np.median([float(r['duration_s']) for r in ok])) if ok else '',
                  '成功样本路径中位数/m':float(np.median([float(r['path_length_m']) for r in ok])) if ok else ''})
    write('numerical/data/60单元汇总.csv',cell)
    fieldmap={'scene_id':'场景编号','seed':'种子','obstacle_count':'障碍物数量','maximum_radius_m':'最大障碍半径/m','method':'方法',
      'outcome_class':'终止分类','reached_goal':'到达目标','collision':'实体碰撞','qp_failures':'QP失败次数','duration_s':'执行时长/s',
      'path_length_m':'已执行路径/m','min_body_clearance_m':'最小实体间隙/m','correction_integral':'累计控制修正/m²·s⁻³',
      'execution_path':'计时执行路径','control_dt':'控制周期/s'}
    write('numerical/data/6000次逐次结果_中文.csv', [{fieldmap[k]: LABELS[METHODS.index(v)] if k=='method' else names[outcomes.index(v)] if k=='outcome_class' else v for k,v in r.items()} for r in data])
    common={}
    for r in data:
        if r['outcome_class']=='goal':common.setdefault(r['scene_id'],set()).add(r['method'])
    common={s for s,m in common.items() if len(m)==4}
    fig,axs=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    ns=[]
    for a,matched in zip(axs,[False,True]):
        nn=[]
        for ni,n in enumerate(COUNTS):
            sizes=[]
            for mi,m in enumerate(METHODS):
                vals=[float(r['correction_integral']) for r in data if r['method']==m and int(r['obstacle_count'])==n and r['outcome_class']=='goal' and (not matched or r['scene_id'] in common)]
                assert vals and min(vals)>0
                sizes.append(len(vals));pos=ni+(mi-1.5)*.19
                if len(vals)==1:a.plot(pos,vals[0],'o',color=COLORS[mi],markersize=7)
                else:
                    a.boxplot([vals],positions=[pos],widths=.16,patch_artist=True,showmeans=True,meanline=True,whis=1.5,
                      boxprops={'facecolor':COLORS[mi],'edgecolor':COLORS[mi],'alpha':.9},
                      medianprops={'color':'black','linewidth':1.4},meanprops={'color':'black','linestyle':':','linewidth':1.6},
                      whiskerprops={'color':COLORS[mi]},capprops={'color':COLORS[mi]},
                      flierprops={'marker':'.','markersize':1.7,'alpha':.24,'markeredgecolor':COLORS[mi]})
            nn.append('/'.join(map(str,sizes)))
        a.set_yscale('log');a.set_xticks(range(5));a.set_xticklabels([f'{n}\n{k}' for n,k in zip(COUNTS,nn)])
        a.set_ylabel('累计控制修正 J（m²/s³）');a.grid(axis='y',alpha=.22);a.set_xlim(-.6,4.6)
        a.set_title('四种方法均成功的相同场景' if matched else '各方法自身成功的运行')
        a.set_xlabel('障碍物数量；下行样本数按图例顺序列出')
        ns.append(nn)
    fig.legend([Patch(facecolor=c) for c in COLORS]+[Line2D([0],[0],color='black'),Line2D([0],[0],color='black',linestyle=':')],LABELS+['中位数','均值'],loc='outside upper center',ncol=6)
    save(fig,'numerical/figures/控制干预箱线图')
    return {'records':6000,'common_success_by_count':ns[1],'own_success_by_count':ns[0]}

def c03():
    ms=['full','no_detour','no_reselection','no_progress_guard','fixed_radius','no_transverse','fixed_strength']
    labels=['完整方法','关闭主动绕行','关闭后续重选','恢复旧评分与保持','固定安全半径','关闭横向项','固定强度请求']
    fig,ax=plt.subplots(figsize=(10,4.8),layout='constrained')
    for i,(m,label) in enumerate(zip(ms,labels)):
        with np.load(ROOT/f'numerical/data/c03/{m}.npz',allow_pickle=False) as z:
            state=z['state'];dist=np.linalg.norm(state[:,1:4]-z['goal'],axis=1)
        ax.plot(state[:,0],dist,label=label,linewidth=2 if i<3 else 1.2,linestyle=['-','--','-.',':','--','-.',':'][i])
    ax.axhline(.25,color='gray',linestyle=':',linewidth=1)
    ax.set(xlabel='时间（s）',ylabel='距离目标（m）',title='C03堵塞场景：只使用已保存轨迹，不延长失败运行',xlim=(0,40),ylim=(0,None))
    ax.grid(alpha=.2);ax.legend(ncol=2,fontsize=9)
    save(fig,'numerical/figures/C03目标推进')

def rotorpy():
    rr=load('rotorpy/data/六次动力学结果.csv')
    four=[r for r in rr if int(r['obstacles'])==4]
    labels=[r['method'].replace('Park-inspired','Park适配') for r in four]
    fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,k,title in zip(axs,['arrival_s','minimum_body_clearance_m','tracking_peak_mps2'],['首次到达时间（s）','最小实体间隙（m）','加速度跟踪误差峰值（m/s²）']):
        vals=[float(r[k]) for r in four]
        ax.bar(range(4),vals,color=['#219D89','#CE5078','#E7A521','#253D94'])
        ax.set_xticks(range(4));ax.set_xticklabels(labels,rotation=20,ha='right');ax.set_title(title,fontsize=11);ax.set_ylim(0,max(vals)*1.23)
        for i,v in enumerate(vals):ax.text(i,v+max(vals)*.025,f'{v:.3f}',ha='center',fontsize=9)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,'rotorpy/figures/四方法动力学指标')
    fig,axs=plt.subplots(2,1,figsize=(10,6.5),layout='constrained',sharex=True)
    for m,label,c in zip(['ra','c3bf','ma_cbf_vo','park'],['RA-DPCBF','C3BF','MA-CBF-VO','Park适配'],COLORS):
        with np.load(ROOT/f'rotorpy/data/four_{m}_history.npz',allow_pickle=False) as z:
            h=z['history'];columns=list(z['columns']);goal=z['goal']
            dist=np.linalg.norm(h[:,1:4]-goal,axis=1)
            err=np.linalg.norm(h[:,[columns.index(k) for k in ['actual_ax','actual_ay','actual_az']]]-h[:,[columns.index(k) for k in ['ux','uy','uz']]],axis=1)
        axs[0].plot(h[:,0],dist,label=label,color=c)
        axs[1].plot(h[:,0],err,label=label,color=c)
    axs[0].set(ylabel='距离目标（m）',title='RotorPy四方法：正式控制记录中的目标推进与跟踪');axs[0].legend(ncol=4)
    axs[1].set(xlabel='时间（s）',ylabel='实际与指令加速度差（m/s²）')
    for a in axs:a.grid(alpha=.2)
    save(fig,'rotorpy/figures/目标推进与跟踪')
    write('rotorpy/data/六次动力学结果_中文.csv',[{'障碍物数量':r['obstacles'],'方法':r['method'].replace('Park-inspired','Park适配'),
       '首次到达时间/s':r['arrival_s'],'路径长度/m':r['path_m'],'最小实体间隙/m':r['minimum_body_clearance_m'],
       '跟踪误差峰值/m·s⁻²':r['tracking_peak_mps2'],'实际屏障最小残差':r['actual_CBF_minimum'],
       '到达后两秒稳定': '是' if r['settled_after_2s']=='True' else '否','实体碰撞':'是' if r['collision']=='True' else '否'} for r in rr])

def px4():
    jj=json.loads((ROOT/'px4-gazebo/data/results.json').read_text(encoding='utf8'));rr=jj['runs']
    methods=['ra','park','ma_cbf_vo','c3bf'];labels=['RA-DPCBF','Park适配','MA-CBF-VO','C3BF']
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');xx=np.arange(4);summary=[]
    for offset,flag,label,color in [(-.18,'arrived','到达目标','#6595D4'),(.18,'strict_execution_passed','全部执行检查通过','#25418F')]:
        vals=[sum(bool(r[flag]) for r in rr if r['method']==m) for m in methods]
        ax.bar(xx+offset,vals,width=.32,label=label,color=color)
        for x,v in zip(xx+offset,vals):ax.text(x,v+.35,f'{v}/25',ha='center',fontsize=10)
    for m,label in zip(methods,labels):
        g=[r for r in rr if r['method']==m];assert len(g)==25
        summary.append({'方法':label,'运行次数':25,'到达次数':sum(r['arrived'] for r in g),'全部执行检查通过次数':sum(r['strict_execution_passed'] for r in g)})
    assert [(r['到达次数'],r['全部执行检查通过次数']) for r in summary]==[(22,16),(12,7),(15,0),(0,0)]
    ax.set(xticks=xx,xticklabels=labels,ylim=(0,28),ylabel='运行次数',title='PX4＋Gazebo：100次运行，到达与执行检查分别统计')
    ax.legend(loc='upper right');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    save(fig,'px4-gazebo/figures/100次到达与执行检查')
    write('px4-gazebo/data/100次中文汇总.csv',summary)
    return summary

def main():
    setup(); receipt={'数值统计':numerical(),'PX4汇总':px4()};c03();rotorpy()
    receipt['数据口径']={'干预量':'J=ΣΔt||u−u_MPC||²；仅成功样本，另列共同成功配对；不是CPU耗时',
      '箱线图':'四分位25%—75%；须为1.5倍IQR；保留全部离群点；n=1只画点；对数轴未删除非正值，断言全部为正',
      'C03':'读取紧凑原状态，逐点计算目标距离；不平滑、不外推、不重跑',
      '结果分类':'数值失败与QP不可行分开；不把超时当碰撞'}
    (ROOT/'reproducibility/中文图表核验.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf8')
    print(json.dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
