import json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
  'font.size':12,'axes.titlesize':13,'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
  'legend.fontsize':11,'axes.grid':True,'grid.color':'#d9d9d9','grid.linewidth':0.6,'grid.linestyle':'--',
  'savefig.dpi':300,'savefig.bbox':'tight','figure.dpi':150})
CB,CT,CO,CG='#2166ac','#5aa1d6','#d6604d','#1a7837'
OUT='figures'

# ---- Variability (Q6): throughput mean +/- 95% CI across seeds ----
D=json.load(open('multiseed_results.json')); seeds=sorted(D,key=int); n=len(seeds)
tc=stats.t.ppf(0.975,n-1)
strats=['Baseline10','Baseline20','DYNASHARD','IntelliShard++']; names=['Base-10','Base-20','Reactive','IS++']
cols=[CB,CT,CO,CG]
fig,ax=plt.subplots(1,2,figsize=(11,4.4)); plt.subplots_adjust(wspace=0.28)
means=[];cis=[]
for st in strats:
    v=np.array([D[s][st]['avg_throughput'] for s in seeds]); means.append(v.mean()); cis.append(tc*v.std(ddof=1)/np.sqrt(n))
b=ax[0].bar(range(4),means,yerr=cis,capsize=6,color=cols,edgecolor='black',lw=0.9,
            error_kw=dict(ecolor='black',lw=1.3),zorder=3)
for bi,m,c in zip(b,means,cis): ax[0].text(bi.get_x()+bi.get_width()/2,m+c+30,f'{m:.0f}',ha='center',fontsize=10.5,fontweight='bold')
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(names); ax[0].set_ylabel('Aggregate TPS')
ax[0].set_title(f'Throughput mean +/- 95% CI ({n} seeds)',fontsize=12.5,fontweight='bold'); ax[0].set_ylim(0,2700)
# latency
meansL=[];cisL=[]
for st in strats:
    v=np.array([D[s][st]['avg_latency'] for s in seeds]); meansL.append(v.mean()); cisL.append(tc*v.std(ddof=1)/np.sqrt(n))
b=ax[1].bar(range(4),meansL,yerr=cisL,capsize=6,color=cols,edgecolor='black',lw=0.9,
            error_kw=dict(ecolor='black',lw=1.3),zorder=3)
ax[1].set_yscale('log')
for bi,m in zip(b,meansL): ax[1].text(bi.get_x()+bi.get_width()/2,m*1.15,f'{m:.1f}',ha='center',fontsize=10.5,fontweight='bold')
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(names); ax[1].set_ylabel('Mean latency (log)')
ax[1].set_title('Latency mean +/- 95% CI (log scale)',fontsize=12.5,fontweight='bold')
fig.savefig(f'{OUT}/fig_variability.png',facecolor='white'); plt.close(fig); print('fig_variability')

# ---- Synergy (Q2) ----
S=json.load(open('q2_synergy.json'))
fig,ax=plt.subplots(figsize=(7,4.4))
labels=['Neither\n(k=10)','RL-only\n(no PSO)','PSO-only\n(no RL)','Full\nRL+PSO']
vals=[S['neither'],S['rl_only'],S['pso_only'],S['full']]; c=[ '#999999',CB,CO,CG]
b=ax.bar(range(4),vals,color=c,edgecolor='black',lw=0.9,zorder=3)
for bi,v in zip(b,vals): ax.text(bi.get_x()+bi.get_width()/2,v+25,f'{v:.0f}',ha='center',fontsize=11,fontweight='bold')
ax.axhline(S['neither'],color='#555',ls='--',lw=1.1)
ax.set_xticks(range(4)); ax.set_xticklabels(labels); ax.set_ylabel('Aggregate TPS')
ax.set_title('RL-PSO synergy: full system exceeds sum of parts',fontsize=12.5,fontweight='bold'); ax.set_ylim(0,2600)
fig.savefig(f'{OUT}/fig_synergy.png',facecolor='white'); plt.close(fig); print('fig_synergy')

# ---- Shard-count curve (Q3) ----
Q3=json.load(open('q3_shardcount.json')); ks=sorted(Q3,key=int)
tps=[Q3[k]['tps'] for k in ks]; persh=[Q3[k]['persh'] for k in ks]
fig,ax=plt.subplots(figsize=(7,4.4)); ax2=ax.twinx()
ax.plot([int(k) for k in ks],tps,color=CG,marker='o',lw=2.2,label='Aggregate TPS')
ax2.plot([int(k) for k in ks],persh,color=CO,marker='s',ls='--',lw=2.0,label='Per-shard TPS')
ax.set_xlabel('Fixed shard count k'); ax.set_ylabel('Aggregate TPS',color=CG); ax2.set_ylabel('Per-shard TPS',color=CO)
ax.set_title('Throughput vs shard count: gains with diminishing returns',fontsize=12.5,fontweight='bold')
ax.grid(True); ax2.grid(False)
l1,la1=ax.get_legend_handles_labels(); l2,la2=ax2.get_legend_handles_labels()
ax.legend(l1+l2,la1+la2,loc='center right',fontsize=10)
fig.savefig(f'{OUT}/fig_shardcurve.png',facecolor='white'); plt.close(fig); print('fig_shardcurve')

# ---- Cost-weight trade-off (Q4) ----
Q4=json.load(open('q4_costweight.json')); ws=sorted(Q4,key=float)
rc=[Q4[w]['reconf'] for w in ws]; lat=[Q4[w]['lat'] for w in ws]
fig,ax=plt.subplots(figsize=(7,4.4)); ax2=ax.twinx()
ax.bar([float(w) for w in ws],rc,width=0.06,color=CB,edgecolor='black',lw=0.8,zorder=3,label='Reconfigurations')
ax2.plot([float(w) for w in ws],lat,color=CO,marker='o',lw=2.2,label='Mean latency')
ax.set_xlabel('Reconfiguration cost weight wR'); ax.set_ylabel('Reconfigurations',color=CB)
ax2.set_ylabel('Mean latency',color=CO)
ax.set_title('Reconfiguration cost weight trade-off',fontsize=12.5,fontweight='bold'); ax2.grid(False)
fig.savefig(f'{OUT}/fig_costweight.png',facecolor='white'); plt.close(fig); print('fig_costweight')

# ---- Reward correlation (Q1) ----
rew=np.array([D[s]['IntelliShard++']['rl_avg_reward'] for s in seeds])
tps=np.array([D[s]['IntelliShard++']['avg_throughput'] for s in seeds])
lat=np.array([D[s]['IntelliShard++']['avg_latency'] for s in seeds])
fig,ax=plt.subplots(1,2,figsize=(11,4.2)); plt.subplots_adjust(wspace=0.28)
ax[0].scatter(rew,tps,color=CG,s=55,edgecolor='black',zorder=3)
zr=np.polyfit(rew,tps,1); xs=np.linspace(rew.min(),rew.max(),50)
ax[0].plot(xs,np.polyval(zr,xs),color='#555',ls='--')
r1,_=stats.pearsonr(rew,tps); ax[0].set_xlabel('Mean RL reward'); ax[0].set_ylabel('Aggregate TPS')
ax[0].set_title(f'Reward vs throughput (r = {r1:+.2f})',fontsize=12.5,fontweight='bold')
ax[1].scatter(rew,lat,color=CO,s=55,edgecolor='black',zorder=3)
zl=np.polyfit(rew,lat,1); ax[1].plot(xs,np.polyval(zl,xs),color='#555',ls='--')
r2,_=stats.pearsonr(rew,lat); ax[1].set_xlabel('Mean RL reward'); ax[1].set_ylabel('Mean latency')
ax[1].set_title(f'Reward vs latency (r = {r2:+.2f})',fontsize=12.5,fontweight='bold')
fig.savefig(f'{OUT}/fig_reward_corr.png',facecolor='white'); plt.close(fig); print('fig_reward_corr')
print('all R3 figures done')
