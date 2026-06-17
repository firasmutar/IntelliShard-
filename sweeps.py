import json, time, numpy as np
import run_sim
from run_sim import run_intellishard, run_baseline

# ---- Q4: reconfiguration-cost-weight sweep ----
print('Q4: reconfiguration cost weight (wR) sweep, N=15000, seed 42')
q4={}
for wR in [0.0, 0.1, 0.3, 0.6, 1.0]:
    run_sim.WEIGHTS['wR']=wR
    r=run_intellishard(15000,42)
    q4[wR]=dict(reconf=r['reconfigurations'], lat=r['avg_latency'], tps=r['avg_throughput'],
                shards=r['avg_n_shards'])
    print(f'  wR={wR}: reconf={r["reconfigurations"]:,}  lat={r["avg_latency"]:.2f}  '
          f'tps={r["avg_throughput"]:.0f}  shards={r["avg_n_shards"]:.2f}',flush=True)
run_sim.WEIGHTS['wR']=0.3  # restore
json.dump(q4,open('q4_costweight.json','w'),indent=1)

# ---- Q3: fixed shard-count vs throughput ----
print('\nQ3: fixed shard count k vs throughput, N=15000, seed 42')
q3={}
for k in [4,6,8,10,12,14,16,18,20]:
    r=run_baseline(15000,42,k=k)
    q3[k]=dict(tps=r['avg_throughput'], lat=r['avg_latency'], persh=r['tps_per_shard'])
    print(f'  k={k:>2}: tps={r["avg_throughput"]:.0f}  lat={r["avg_latency"]:.2f}  '
          f'per-shard={r["tps_per_shard"]:.0f}',flush=True)
json.dump(q3,open('q3_shardcount.json','w'),indent=1)
print('DONE')
