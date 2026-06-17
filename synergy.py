import json, numpy as np
import run_sim
from run_sim import (EthereumWorkload, ShardSystem, AGBWPSO, QAgent, ACTIONS,
                     objective, RECONF_COST, WEIGHTS, util_std, Accumulator, run_baseline)

def run_rl_only(N, seed):
    """Q-learning chooses actions, but rebalance uses uniform max capacity (no PSO)."""
    wl=EthereumWorkload(seed=seed); rng=np.random.default_rng(seed+1)
    sysm=ShardSystem(k=10,rng=rng); agent=QAgent(rng,eps_decay_steps=int(0.6*N),k_min=4,k_max=20)
    acc=Accumulator(); reconf_total=0; prev_s=prev_a=None; prev_pot=0.0
    for _ in range(N):
        w=wl.next_window(sysm.k); ustd,umean,umax=util_std(sysm,w)
        tr=np.mean([np.mean(t) for t in sysm.trust])
        unmet=max(0.0,1.0-1.0/max(umean,1e-6)) if umean>1 else 0.0
        s=agent.encode(dict(mean_util=umean,max_util=umax,util_std=ustd,trust=tr,rho_c=w['cross_shard_ratio'],k=sysm.k,unmet=unmet))
        a=agent.act(s,sysm.k); action=ACTIONS[a]; reconf=0
        if action=='split':
            tx=w['shard_tx']; tx=np.resize(tx,sysm.k) if len(tx)!=sysm.k else tx
            reconf=sysm.split(int(np.argmax(tx/np.maximum(sysm.capacity,1e-9))))
        elif action=='merge':
            o=np.argsort(sysm.capacity)
            if sysm.k>2: reconf=sysm.merge(int(o[0]),int(o[1]))
        elif action=='rebalance':
            sysm.set_capacity(np.full(sysm.k,350.0)); reconf=1   # uniform max, NO PSO
        reconf_total+=reconf; m=sysm.evaluate(w)
        sr,ls,ov=m['served_ratio'],m['latency_score'],m['overload_frac']
        pot=sr; shaping=agent.gamma*pot-prev_pot; prev_pot=pot
        r=3*sr+1*ls-2.5*ov-WEIGHTS['wR']*(RECONF_COST[action]/100.0)+1*shaping
        s2=agent.encode(dict(mean_util=m['mean_util'],max_util=m['max_util'],util_std=ustd,trust=tr,rho_c=w['cross_shard_ratio'],k=sysm.k,unmet=m['unmet_ratio']))
        if prev_s is not None: agent.update(prev_s,prev_a,r,s)
        prev_s,prev_a=s,a; agent.step(); acc.add(m,reconf=0)
    acc.reconf=reconf_total; return acc.summary()

def run_pso_only(N, seed):
    """No Q-learning: every 10 windows run PSO and adjust to k*; else do nothing."""
    wl=EthereumWorkload(seed=seed); rng=np.random.default_rng(seed+1)
    sysm=ShardSystem(k=10,rng=rng); pso=AGBWPSO(rng); acc=Accumulator(); reconf_total=0; cached=None; step=0
    for _ in range(N):
        w=wl.next_window(sysm.k); step+=1; reconf=0
        if step%10==1:
            total=float(np.sum(w['shard_tx']))
            def fit(x):
                k=int(x[0]); caps=np.clip(x[1:1+k],1e2,350.0)
                if len(caps)<k: caps=np.pad(caps,(0,k-len(caps)),constant_values=300.0)
                wts=rng.dirichlet(np.full(k,0.8)); dem=wts*total; util=dem/np.maximum(caps,1e-9)
                L=float(np.mean(1.0/np.maximum(1.0-np.minimum(util,0.98),0.02)))
                E=float(np.sum(caps)*0.05)/100.0; T=float(np.sum(np.minimum(dem,caps)))/100.0
                C=(1-0.4/k)*total*(k**0.5)*0.02/10.0
                return objective(L,E,T,C,RECONF_COST['rebalance'])
            ks,caps,gf,h=pso.optimise(fit,total); cached=(ks,caps)
        if cached:
            ks,caps=cached; g=0
            while sysm.k<ks and sysm.k<20 and g<30:
                if sysm.split(int(np.argmax(sysm.capacity)))==0: break
                reconf+=1; g+=1
            g=0
            while sysm.k>ks and sysm.k>2 and g<30:
                o=np.argsort(sysm.capacity)
                if sysm.merge(int(o[0]),int(o[1]))==0: break
                reconf+=1; g+=1
            if len(caps)==sysm.k: sysm.set_capacity(caps)
        reconf_total+=reconf; m=sysm.evaluate(w); acc.add(m,reconf=0)
    acc.reconf=reconf_total; return acc.summary()

N=15000; seed=42
print('Q2 synergy study, N=%d, seed=%d'%(N,seed))
neither=run_baseline(N,seed,k=10)
rl=run_rl_only(N,seed)
ps=run_pso_only(N,seed)
full=run_sim.run_intellishard(N,seed)
res={'neither':neither['avg_throughput'],'rl_only':rl['avg_throughput'],
     'pso_only':ps['avg_throughput'],'full':full['avg_throughput']}
for k,v in [('Neither (fixed k=10)',neither),('RL-only (no PSO)',rl),('PSO-only (no RL)',ps),('Full RL+PSO',full)]:
    print(f'  {k:<22} tps={v["avg_throughput"]:.0f}  lat={v["avg_latency"]:.2f}  shards={v["avg_n_shards"]:.2f}')
base=neither['avg_throughput']
gain_rl=rl['avg_throughput']-base; gain_ps=ps['avg_throughput']-base; gain_full=full['avg_throughput']-base
print(f'\n  gain RL-only:  {gain_rl:+.0f} TPS')
print(f'  gain PSO-only: {gain_ps:+.0f} TPS')
print(f'  sum of parts:  {gain_rl+gain_ps:+.0f} TPS')
print(f'  full combined: {gain_full:+.0f} TPS')
print(f'  synergy (full - sum of parts): {gain_full-(gain_rl+gain_ps):+.0f} TPS')
json.dump({'neither':neither['avg_throughput'],'rl_only':rl['avg_throughput'],'pso_only':ps['avg_throughput'],
           'full':full['avg_throughput'],'lat':{'neither':neither['avg_latency'],'rl_only':rl['avg_latency'],
           'pso_only':ps['avg_latency'],'full':full['avg_latency']}},open('q2_synergy.json','w'),indent=1)
