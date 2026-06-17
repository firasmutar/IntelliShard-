"""
run_sim.py  -  Main driver. Runs Baseline, DYNASHARD, IntelliShard++ over N
windows on the SAME workload seed, computes aggregate metrics, writes results.json.

Whatever this prints is the real result. No targets, no hardcoding.
"""
import json, sys, time
import numpy as np
from workload import EthereumWorkload
from shard_model import ShardSystem
from pso import AGBWPSO
from qlearn import QAgent, ACTIONS

WEIGHTS = dict(wL=1.0, wE=1.0, wT=1.0, wC=1.0, wR=0.3)
RECONF_COST = {"do_nothing": 0.0, "split": 50.0, "merge": 50.0, "rebalance": 100.0}


def objective(L, E, T, C, R):
    return (WEIGHTS["wL"] * L + WEIGHTS["wE"] * E
            - WEIGHTS["wT"] * T + WEIGHTS["wC"] * C + WEIGHTS["wR"] * R)


def util_std(sysm, window):
    tx = window["shard_tx"]
    if len(tx) != sysm.k:
        tx = np.resize(tx, sysm.k)
    u = tx / np.maximum(sysm.capacity, 1e-9)
    return float(np.std(u)), float(np.mean(u)), float(np.max(u))


def run_baseline(N, seed, k=10):
    wl = EthereumWorkload(seed=seed)
    rng = np.random.default_rng(seed + 1)
    sysm = ShardSystem(k=k, rng=rng)
    acc = Accumulator()
    for _ in range(N):
        w = wl.next_window(sysm.k)
        m = sysm.evaluate(w)
        acc.add(m, reconf=0)
    return acc.summary()


def run_dynashard(N, seed):
    wl = EthereumWorkload(seed=seed)
    rng = np.random.default_rng(seed + 1)
    sysm = ShardSystem(k=10, rng=rng)
    acc = Accumulator()
    reconf_total = 0
    for _ in range(N):
        w = wl.next_window(sysm.k)
        m = sysm.evaluate(w)
        # reactive: split overloaded, merge underused. NO cooldown, NO cost.
        # recompute utilisation against the CURRENT topology each time.
        def cur_util():
            tx = w["shard_tx"]
            if len(tx) != sysm.k:
                tx = np.resize(tx, sysm.k)
            return tx / np.maximum(sysm.capacity, 1e-9)
        # split: re-evaluate after each split because k changes (cascading)
        guard = 0
        while guard < 64:
            u = cur_util()
            over = np.where(u > 0.80)[0]
            if len(over) == 0 or sysm.k >= 20:
                break
            reconf_total += sysm.split(int(over[0]))
            guard += 1
        # merge: repeatedly join the two least-utilised while both < 0.20
        guard = 0
        while guard < 64 and sysm.k > 2:
            u = cur_util()
            low = np.where(u < 0.20)[0]
            if len(low) < 2:
                break
            order = low[np.argsort(u[low])]
            i, j = int(order[0]), int(order[1])
            reconf_total += sysm.merge(min(i, j), max(i, j))
            guard += 1
        acc.add(m, reconf=0)
    acc.reconf = reconf_total
    return acc.summary()


def run_intellishard(N, seed, eps_horizon=None):
    wl = EthereumWorkload(seed=seed)
    rng = np.random.default_rng(seed + 1)
    sysm = ShardSystem(k=10, rng=rng)
    if eps_horizon is None:
        eps_horizon = int(0.6 * N)            # decay over 60% of the run, then exploit
    agent = QAgent(rng, eps_decay_steps=eps_horizon, k_min=4, k_max=20)
    pso = AGBWPSO(rng)
    acc = Accumulator()
    reconf_total = 0
    rebalance_calls = 0
    cached = None
    prev_s = prev_a = None
    rewards = []
    kstar_hist = []
    pso_conv = None
    action_counts = {a: 0 for a in ACTIONS}
    prev_potential = 0.0

    for _ in range(N):
        w = wl.next_window(sysm.k)
        ustd, umean, umax = util_std(sysm, w)
        trust_now = np.mean([np.mean(t) for t in sysm.trust])
        # peek unmet from a dry evaluate would be circular; use util proxy
        unmet_proxy = max(0.0, 1.0 - 1.0 / max(umean, 1e-6)) if umean > 1 else 0.0
        feat = dict(mean_util=umean, max_util=umax, util_std=ustd,
                    trust=trust_now, rho_c=w["cross_shard_ratio"], k=sysm.k,
                    unmet=unmet_proxy)
        s = agent.encode(feat)
        a = agent.act(s, sysm.k)
        action = ACTIONS[a]
        action_counts[action] += 1

        reconf = 0
        if action == "split":
            # split the most-loaded shard (highest demand/capacity), not highest capacity
            tx = w["shard_tx"]
            if len(tx) != sysm.k:
                tx = np.resize(tx, sysm.k)
            util_now = tx / np.maximum(sysm.capacity, 1e-9)
            idx = int(np.argmax(util_now))
            reconf = sysm.split(idx)
        elif action == "merge":
            order = np.argsort(sysm.capacity)
            if sysm.k > 2:
                reconf = sysm.merge(int(order[0]), int(order[1]))
        elif action == "rebalance":
            rebalance_calls += 1
            if rebalance_calls % 10 == 1:
                total = float(np.sum(w["shard_tx"]))
                def fit(x):
                    k = int(x[0]); caps = np.clip(x[1:1 + k], 1e2, 350.0)
                    if len(caps) < k:
                        caps = np.pad(caps, (0, k - len(caps)), constant_values=300.0)
                    wts = rng.dirichlet(np.full(k, 0.8))
                    dem = wts * total
                    util = dem / np.maximum(caps, 1e-9)
                    L = float(np.mean(1.0 / np.maximum(1.0 - np.minimum(util, 0.98), 0.02)))
                    E = float(np.sum(caps) * 0.05) / 100.0
                    T = float(np.sum(np.minimum(dem, caps))) / 100.0
                    rho = 1.0 - 0.40 / k
                    C = rho * total * (k ** 0.5) * 0.02 / 10.0
                    return objective(L, E, T, C, RECONF_COST["rebalance"])
                k_star, caps, gf, hist = pso.optimise(fit, total)
                cached = (k_star, caps)
                kstar_hist.append(k_star)
                if pso_conv is None:
                    pso_conv = hist
            if cached is not None:
                k_star, caps = cached
                guard = 0
                while sysm.k < k_star and sysm.k < 20 and guard < 30:
                    if sysm.split(int(np.argmax(sysm.capacity))) == 0:
                        break
                    reconf += 1; guard += 1
                guard = 0
                while sysm.k > k_star and sysm.k > 2 and guard < 30:
                    order = np.argsort(sysm.capacity)
                    if sysm.merge(int(order[0]), int(order[1])) == 0:
                        break
                    reconf += 1; guard += 1
                if len(caps) == sysm.k:
                    sysm.set_capacity(caps)
            reconf = max(reconf, 1)

        reconf_total += reconf
        m = sysm.evaluate(w)

        # ---- BOUNDED, NORMALISED REWARD (all terms in comparable ranges) ----
        served_ratio = m["served_ratio"]          # 0..1  (want high)
        lat_score    = m["latency_score"]          # 0..1  (want high)
        overload_pen = m["overload_frac"]          # 0..1  (want low)
        # reconfiguration cost normalised to a small penalty
        reconf_pen   = RECONF_COST[action] / 100.0 # 0..1
        # potential-based shaping: reward increase in served_ratio
        potential = served_ratio
        shaping = agent.gamma * potential - prev_potential
        prev_potential = potential

        r = (3.0 * served_ratio
             + 1.0 * lat_score
             - 2.5 * overload_pen
             - WEIGHTS["wR"] * reconf_pen
             + 1.0 * shaping)
        rewards.append(r)

        feat2 = dict(mean_util=m["mean_util"], max_util=m["max_util"],
                     util_std=ustd, trust=trust_now,
                     rho_c=w["cross_shard_ratio"], k=sysm.k,
                     unmet=m["unmet_ratio"])
        s2 = agent.encode(feat2)
        if prev_s is not None:
            agent.update(prev_s, prev_a, r, s)
        prev_s, prev_a = s, a
        agent.step()

        acc.add(m, reconf=0)

    acc.reconf = reconf_total
    out = acc.summary()
    out["rl_avg_reward"] = float(np.mean(rewards))
    out["rl_reward_curve"] = [float(np.mean(rewards[i:i+max(N//50,1)]))
                              for i in range(0, len(rewards), max(N//50,1))]
    out["kstar_mode"] = int(np.bincount(kstar_hist).argmax()) if kstar_hist else None
    out["kstar_hist"] = kstar_hist
    out["pso_conv"] = pso_conv
    out["action_counts"] = action_counts
    return out


class Accumulator:
    def __init__(self):
        self.keys = ["agg_tps", "mean_lat", "energy", "avg_trust", "consensus",
                     "bft", "srs", "dsp", "cft", "so", "k", "mean_util", "max_util"]
        self.sums = {k: 0.0 for k in self.keys}
        self.lat_min = float("inf"); self.lat_max = 0.0
        self.n_attacks = 0; self.n_detected = 0
        self.n = 0; self.reconf = 0

    def add(self, m, reconf=0):
        for k in self.keys:
            self.sums[k] += m[k]
        self.lat_min = min(self.lat_min, m["mean_lat"])
        self.lat_max = max(self.lat_max, m["mean_lat"])
        self.n_attacks += m["n_attacks"]
        self.n_detected += m["n_detected"]
        self.reconf += reconf
        self.n += 1

    def summary(self):
        n = max(self.n, 1)
        avg = {k: self.sums[k] / n for k in self.keys}
        adr = self.n_detected / max(self.n_attacks, 1)
        return {
            "avg_throughput": avg["agg_tps"],
            "avg_latency": avg["mean_lat"],
            "min_latency": self.lat_min,
            "max_latency": self.lat_max,
            "avg_energy": avg["energy"],
            "avg_n_shards": avg["k"],
            "avg_trust": avg["avg_trust"],
            "avg_consensus": avg["consensus"],
            "bft_ratio": avg["bft"],
            "adr": adr,
            "srs": avg["srs"],
            "dsp": avg["dsp"],
            "cft": avg["cft"],
            "so": avg["so"],
            "total_attacks": self.n_attacks,
            "total_detected": self.n_detected,
            "reconfigurations": self.reconf,
            "tps_per_shard": avg["agg_tps"] / max(avg["k"], 1),
            "energy_per_tps": avg["energy"] / max(avg["agg_tps"], 1),
        }


if __name__ == "__main__":
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    seed = 42
    print(f"Running simulation: N={N} windows, seed={seed}\n", flush=True)
    t0 = time.time()

    print("[1/4] Baseline-10 (fixed 10 shards)...", flush=True)
    base10 = run_baseline(N, seed, k=10)
    print("[2/4] Baseline-20 (fixed 20 shards, well-provisioned)...", flush=True)
    base20 = run_baseline(N, seed, k=20)
    print("[3/4] DYNASHARD (reactive, no cost)...", flush=True)
    dyn = run_dynashard(N, seed)
    print("[4/4] IntelliShard++ (Q-learning + AGBWPSO)...", flush=True)
    iss = run_intellishard(N, seed)

    meta = {"N": N, "seed": seed, "weights": WEIGHTS,
            "runtime_s": round(time.time() - t0, 1)}

    # save full arrays for figures, then strip from json
    np.savez("sim_arrays.npz",
             kstar_hist=np.array(iss.get("kstar_hist") or []),
             pso_conv=np.array(iss.get("pso_conv") or []),
             reward_curve=np.array(iss.get("rl_reward_curve") or []))
    iss_clean = {k: v for k, v in iss.items()
                 if k not in ("kstar_hist", "pso_conv")}

    with open("results.json", "w") as f:
        json.dump({"meta": meta, "Baseline10": base10, "Baseline20": base20,
                   "DYNASHARD": dyn, "IntelliShard++": iss_clean}, f, indent=2)

    print(f"\nDone in {meta['runtime_s']}s. Wrote results.json\n", flush=True)
    print(f"{'Metric':<20}{'Base-10':>11}{'Base-20':>11}{'DYNASHARD':>13}{'IS++':>11}")
    print("-" * 66)
    rows = [("Per-shard TPS", "tps_per_shard", ".1f"),
            ("Aggregate TPS", "avg_throughput", ".0f"),
            ("Mean latency", "avg_latency", ".2f"),
            ("Max latency", "max_latency", ".0f"),
            ("Avg shards", "avg_n_shards", ".2f"),
            ("Reconfigurations", "reconfigurations", ".0f"),
            ("ADR", "adr", ".4f"),
            ("SRS", "srs", ".4f"),
            ("Consensus", "avg_consensus", ".4f"),
            ("BFT ratio", "bft_ratio", ".4f"),
            ("CFT (rounds)", "cft", ".2f"),
            ("SO (ms)", "so", ".4f"),
            ("Trust", "avg_trust", ".2f"),
            ("DSP", "dsp", ".4f"),
            ("Attacks", "total_attacks", ".0f"),
            ("Detected", "total_detected", ".0f")]
    for label, key, fmt in rows:
        vals = [format(d[key], fmt) for d in (base10, base20, dyn, iss)]
        print(f"{label:<20}{vals[0]:>11}{vals[1]:>11}{vals[2]:>13}{vals[3]:>11}")
    print(f"\nIS++ RL avg reward: {iss['rl_avg_reward']:.3f}")
    print(f"IS++ PSO modal k*: {iss['kstar_mode']}")
    print(f"IS++ vs Base-10 TPS: {100*(iss['avg_throughput']/base10['avg_throughput']-1):+.1f}%")
    print(f"IS++ vs Base-20 TPS: {100*(iss['avg_throughput']/base20['avg_throughput']-1):+.1f}%")
    print(f"IS++ vs DYNA latency: {dyn['avg_latency']/iss['avg_latency']:.1f}x better")
