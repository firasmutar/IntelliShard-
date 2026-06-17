import json, os, sys, time
from run_sim import run_baseline, run_dynashard, run_intellishard
N=20000
OUT='multiseed_results.json'
data=json.load(open(OUT)) if os.path.exists(OUT) else {}
seeds=[42,7,13,101,202,303,404,505,606,707]
budget=float(sys.argv[1]) if len(sys.argv)>1 else 240
t0=time.time()
for s in seeds:
    if str(s) in data: continue
    if time.time()-t0>budget: break
    st=time.time()
    rec={}
    for name,fn in [('Baseline10',lambda: run_baseline(N,s,k=10)),
                    ('Baseline20',lambda: run_baseline(N,s,k=20)),
                    ('DYNASHARD',lambda: run_dynashard(N,s)),
                    ('IntelliShard++',lambda: run_intellishard(N,s))]:
        r=fn()
        rec[name]={k:r[k] for k in ('avg_throughput','avg_latency','max_latency','avg_n_shards',
                   'reconfigurations','adr','srs','bft_ratio','avg_consensus','cft','energy_per_tps',
                   'tps_per_shard') if k in r}
        if name=='IntelliShard++':
            rec[name]['rl_avg_reward']=r.get('rl_avg_reward')
            rec[name]['kstar_mode']=r.get('kstar_mode')
    data[str(s)]=rec
    json.dump(data,open(OUT,'w'),indent=1)
    print(f'seed {s} done in {time.time()-st:.0f}s  (total seeds={len(data)})',flush=True)
print('STOP. seeds completed:',sorted(int(k) for k in data))
