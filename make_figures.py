#!/usr/bin/env python3
"""
make_figures.py  -  Generate all paper figures FROM THE REAL results.json.
No hardcoded data: every value is read from the simulator output.
IJIES-compliant: no (a)(b)(c) labels, >=10pt fonts, no outer borders, 300 DPI.
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(HERE, 'results.json')))
A = np.load(os.path.join(HERE, 'sim_arrays.npz'))
META = R['meta']

# strategy keys and display
B10, B20, DYN, ISS = R['Baseline10'], R['Baseline20'], R['DYNASHARD'], R['IntelliShard++']
NAMES = ['Base-10', 'Base-20', 'DYNASHARD', 'IS++']
DATA = [B10, B20, DYN, ISS]
CB, CT, CO, CG = '#2166ac', '#5aa1d6', '#d6604d', '#1a7837'
COLORS = [CB, CT, CO, CG]
HATCH = ['////', '\\\\', 'xx', '....']

plt.rcParams.update({
    'figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white',
    'font.family':'serif','font.serif':['Times New Roman','Times','DejaVu Serif'],
    'font.size':12,'axes.titlesize':13,'axes.labelsize':12,
    'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':11,
    'axes.edgecolor':'#222','axes.linewidth':0.9,
    'xtick.direction':'in','ytick.direction':'in',
    'axes.grid':True,'grid.color':'#d9d9d9','grid.linewidth':0.6,
    'grid.linestyle':'--','grid.alpha':0.8,'lines.linewidth':2.0,'lines.markersize':7,
    'legend.frameon':True,'legend.framealpha':0.95,'legend.edgecolor':'#888',
    'figure.dpi':150,'savefig.dpi':300,'savefig.bbox':'tight','savefig.pad_inches':0.06,
    'mathtext.fontset':'stix',
})

def save(fig, name):
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'  {name}  ({os.path.getsize(p)//1024} KB)')
    plt.close(fig)

def bars(ax, key, ylabel, fmt='.1f', star_best='max', ylog=False):
    vals = [d[key] for d in DATA]
    b = ax.bar(range(4), vals, width=0.62, color=COLORS, hatch=HATCH,
               edgecolor='black', lw=0.9, zorder=3)
    if ylog:
        ax.set_yscale('log')
    mx = max(vals)
    for bi, v in zip(b, vals):
        ax.text(bi.get_x()+bi.get_width()/2, v*(1.02 if ylog else 1.0)+ (0 if ylog else mx*0.02),
                format(v, fmt), ha='center', va='bottom', fontsize=10.5, fontweight='bold')
    if star_best:
        idx = int(np.argmax(vals)) if star_best=='max' else int(np.argmin(vals))
        b[idx].set_linewidth(2.6); b[idx].set_edgecolor('black')
    ax.set_xticks(range(4)); ax.set_xticklabels(NAMES, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    if not ylog:
        ax.set_ylim(0, mx*1.18)
    return vals

# ---- FIG: performance (4 panels) ----
def fig_perf():
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    plt.subplots_adjust(hspace=0.34, wspace=0.26)
    bars(ax[0,0], 'avg_throughput', 'Aggregate TPS', fmt='.0f')
    ax[0,0].set_title('Aggregate throughput (higher better)', fontsize=12.5, fontweight='bold', pad=7)
    bars(ax[0,1], 'tps_per_shard', 'TPS per shard', fmt='.1f')
    ax[0,1].set_title('Per-shard throughput (higher better)', fontsize=12.5, fontweight='bold', pad=7)
    bars(ax[1,0], 'avg_latency', 'Mean latency (units)', fmt='.2f', star_best='min', ylog=True)
    ax[1,0].set_title('Mean latency, log scale (lower better)', fontsize=12.5, fontweight='bold', pad=7)
    bars(ax[1,1], 'avg_n_shards', 'Average active shards', fmt='.2f', star_best=None)
    ax[1,1].set_title('Average active shard count', fontsize=12.5, fontweight='bold', pad=7)
    save(fig, 'fig_performance.png')

# ---- FIG: security (4 panels) ----
def fig_sec():
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.4))
    plt.subplots_adjust(hspace=0.34, wspace=0.26)
    bars(ax[0,0], 'adr', 'ADR', fmt='.4f'); ax[0,0].set_ylim(0.9,1.005)
    ax[0,0].set_title('Attack detection rate (higher better)', fontsize=12.5, fontweight='bold', pad=7)
    bars(ax[0,1], 'bft_ratio', 'BFT ratio', fmt='.4f'); ax[0,1].set_ylim(0.8,1.02)
    ax[0,1].set_title('BFT ratio (higher better)', fontsize=12.5, fontweight='bold', pad=7)
    bars(ax[1,0], 'avg_consensus', 'Consensus score', fmt='.4f'); ax[1,0].set_ylim(0,1.0)
    ax[1,0].set_title('Consensus score (higher better)', fontsize=12.5, fontweight='bold', pad=7)
    bars(ax[1,1], 'cft', 'Finality time (rounds)', fmt='.2f', star_best='min')
    ax[1,1].set_title('Consensus finality time (lower better)', fontsize=12.5, fontweight='bold', pad=7)
    save(fig, 'fig_security.png')

# ---- FIG: latency vs reconfig (the stability story) ----
def fig_stability():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    plt.subplots_adjust(wspace=0.30)
    # reconfigurations (log)
    vals=[d['reconfigurations'] for d in DATA]
    vals=[max(v,0.5) for v in vals]
    b=ax[0].bar(range(4),vals,width=0.62,color=COLORS,hatch=HATCH,edgecolor='black',lw=0.9,zorder=3)
    ax[0].set_yscale('log')
    for bi,v,d in zip(b,vals,DATA):
        ax[0].text(bi.get_x()+bi.get_width()/2,v*1.3,f"{d['reconfigurations']:,}",ha='center',va='bottom',fontsize=10,fontweight='bold')
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(NAMES,fontsize=11)
    ax[0].set_ylabel('Reconfigurations (log)'); ax[0].set_title('Total reconfigurations over 100k windows',fontsize=12.5,fontweight='bold',pad=7)
    # max latency (log) - DYNASHARD instability
    bars(ax[1],'max_latency','Maximum latency (units, log)',fmt='.0f',star_best='min',ylog=True)
    ax[1].set_title('Worst-case latency (lower better)',fontsize=12.5,fontweight='bold',pad=7)
    save(fig,'fig_stability.png')

# ---- FIG: RL learning curve + k* distribution ----
def fig_learning():
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    plt.subplots_adjust(wspace=0.28)
    rc = A['reward_curve']
    x = np.linspace(0, META['N']/1000, len(rc))
    ax[0].plot(x, rc, color=CG, lw=2.2, marker='o', markersize=3)
    # smoothed trend
    if len(rc) >= 5:
        sm = np.convolve(rc, np.ones(5)/5, mode='valid')
        ax[0].plot(x[2:-2], sm, color='#c0392b', lw=1.6, ls='--', label='5-point trend')
        ax[0].legend(fontsize=10)
    ax[0].set_xlabel('Training step (x1000)'); ax[0].set_ylabel('Mean reward')
    ax[0].set_title('Q-learning reward over training', fontsize=12.5, fontweight='bold', pad=7)
    # k* distribution
    ks = A['kstar_hist']
    if len(ks):
        vc = np.bincount(ks, minlength=21)
        kk = np.arange(2,21)
        ax[1].bar(kk, vc[2:21], width=0.8, color='white', hatch='////', edgecolor='black', lw=0.9, zorder=3)
        mode = int(np.argmax(vc))
        ax[1].axvline(mode, color=CG, ls='--', lw=2.2, label=f'modal k*={mode}')
        ax[1].legend(fontsize=11)
    ax[1].set_xlabel('PSO-optimal shard count k*'); ax[1].set_ylabel('Frequency')
    ax[1].set_title('Distribution of PSO-selected k*', fontsize=12.5, fontweight='bold', pad=7)
    save(fig, 'fig_learning.png')

# ---- FIG: PSO convergence ----
def fig_pso():
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    pc = A['pso_conv']
    ax.plot(range(len(pc)), pc, color=CG, lw=2.2, marker='o', markersize=6)
    ax.fill_between(range(len(pc)), pc, np.min(pc), alpha=0.10, color=CG)
    ax.set_xlabel('PSO iteration'); ax.set_ylabel('Best fitness f(X)')
    ax.set_title('AGBWPSO fitness convergence (sample rebalance call)', fontsize=12.5, fontweight='bold', pad=7)
    save(fig, 'fig_pso.png')

# ---- FIG: improvement summary ----
def fig_summary():
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    gains = [
        ('TPS vs\nBase-10', 100*(ISS['avg_throughput']/B10['avg_throughput']-1)),
        ('TPS vs\nBase-20', 100*(ISS['avg_throughput']/B20['avg_throughput']-1)),
        ('Latency vs\nDYNASHARD', 100*(1-ISS['avg_latency']/DYN['avg_latency'])),
        ('CFT vs\nBase-10', 100*(1-ISS['cft']/B10['cft'])),
        ('Consensus vs\nBase-10', 100*(ISS['avg_consensus']/B10['avg_consensus']-1)),
    ]
    labels=[g[0] for g in gains]; vals=[g[1] for g in gains]
    cols=[CG if v>=0 else CO for v in vals]
    b=ax.bar(range(len(vals)),vals,width=0.6,color=cols,edgecolor='black',lw=0.9,zorder=3)
    for bi,v in zip(b,vals):
        ax.text(bi.get_x()+bi.get_width()/2, v+(1 if v>=0 else -3), f'{v:+.1f}%',ha='center',va='bottom' if v>=0 else 'top',fontsize=10.5,fontweight='bold')
    ax.axhline(0,color='black',lw=0.8)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels,fontsize=10)
    ax.set_ylabel('Improvement (%)')
    ax.set_title('IntelliShard++ improvements (real, 100k windows)',fontsize=12.5,fontweight='bold',pad=7)
    save(fig,'fig_summary.png')

print('Generating figures from REAL results.json (N=%d)...' % META['N'])
fig_perf(); fig_sec(); fig_stability(); fig_learning(); fig_pso(); fig_summary()
print('Done. Figures in', OUT)
