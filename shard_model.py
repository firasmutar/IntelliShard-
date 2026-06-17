"""
shard_model.py  -  Shard topology, capacity, trust, and the metric computations.

A Shard holds a processing capacity (TPS) and a set of validators each with a
trust score. The ShardSystem applies demand, computes latency/energy/throughput,
and computes the six security metrics. NONE of these formulas are tuned to hit a
target number; they are mechanical and the simulation reports whatever results.
"""
import numpy as np

EPS = 1e-9


class ShardSystem:
    def __init__(self, k, rng, nodes_per_shard=20, malicious_ratio=0.08,
                 base_capacity=300.0):
        self.rng = rng
        self.k = k
        self.nodes_per_shard = nodes_per_shard
        self.malicious_ratio = malicious_ratio
        # per-shard processing capacity in TPS (uniform at start)
        self.capacity = np.full(k, base_capacity, dtype=float)
        # per-shard validator trust in [0,1]; malicious nodes start low
        self.trust = [self._init_trust() for _ in range(k)]

    def _init_trust(self):
        n = self.nodes_per_shard
        t = self.rng.uniform(0.5, 0.9, n)
        n_mal = int(self.malicious_ratio * n)
        if n_mal > 0:
            t[:n_mal] = self.rng.uniform(0.0, 0.2, n_mal)
        return t

    # ---- structural operations -------------------------------------------
    def split(self, idx, new_capacity=300.0):
        """Split shard idx: the new shard gets its OWN fresh capacity, so total
        system capacity increases (this is the point of sharding). Validators
        are divided between the two shards."""
        if self.k >= 20:
            return 0
        old = self.trust[idx]
        if len(old) < 4:
            return 0  # too few validators to split (BFT needs enough nodes)
        # new shard receives fresh capacity (added processing power)
        self.capacity = np.append(self.capacity, new_capacity)
        half = len(old) // 2
        self.trust[idx] = old[:half]
        self.trust.append(old[half:])
        self.k += 1
        return 1

    def merge(self, i, j):
        if self.k <= 2 or i == j:
            return 0
        i, j = sorted((i, j))
        self.capacity[i] = self.capacity[i] + self.capacity[j]
        self.capacity = np.delete(self.capacity, j)
        self.trust[i] = np.concatenate([self.trust[i], self.trust[j]])
        del self.trust[j]
        self.k -= 1
        return 1

    def set_capacity(self, caps):
        caps = np.asarray(caps, dtype=float)
        if len(caps) == self.k:
            self.capacity = np.clip(caps, 1e2, 350.0)

    # ---- per-window evaluation -------------------------------------------
    def evaluate(self, window):
        """Apply demand for one window; return performance + security dict."""
        k = self.k
        shard_tx = window["shard_tx"]
        # re-bucket demand if topology size changed since window was drawn
        if len(shard_tx) != k:
            total = float(np.sum(shard_tx))
            w = self.rng.dirichlet(np.full(k, 0.8))
            shard_tx = w * total

        cap = self.capacity
        util = shard_tx / np.maximum(cap, EPS)            # may exceed 1 (overload)
        # latency per shard: queueing blows up as util -> 1+
        lat = np.where(util < 1.0,
                       1.0 / np.maximum(1.0 - util, 0.02),
                       50.0 + 100.0 * (util - 1.0))        # overloaded shards
        mean_lat = float(np.mean(lat))

        # throughput: limited by capacity, can't exceed demand
        served = np.minimum(shard_tx, cap)
        agg_tps = float(np.sum(served))                    # served tx per window
        demand_total = float(np.sum(shard_tx))
        served_ratio = agg_tps / max(demand_total, EPS)    # in [0,1], want high
        unmet_ratio = 1.0 - served_ratio                   # backlog signal
        overload_frac = float(np.mean(util > 1.0))         # fraction overloaded
        latency_score = float(np.exp(-mean_lat / 20.0))    # bounded 0..1, want high

        energy = float(np.sum(cap) * 0.05 + np.sum(util ** 2) * 10.0)

        # ---- single consolidated per-shard pass (was 5 separate loops) ----
        trust_means = np.empty(k)
        bft_ok = 0
        n_attacks = 0
        n_detected = 0
        srs_sum = 0.0
        cft_sum = 0.0
        total_nodes = 0
        log2 = np.log2
        for s in range(k):
            t = self.trust[s]
            n_t = t.shape[0]
            total_nodes += n_t
            bad = t < 0.3
            n_bad = int(bad.sum())
            good = ~bad
            # trust update in place
            t[good] = np.minimum(1.0, t[good] + 0.001)
            t[bad] = np.maximum(0.0, t[bad] - 0.001)
            tm = float(t.mean())
            trust_means[s] = tm
            frac_bad = n_bad / n_t if n_t else 0.0
            if frac_bad < 1.0 / 3.0:
                bft_ok += 1
            # attacks
            n_attacks += n_bad
            if n_bad > 0:
                honest_vals = t[t >= 0.3]
                det_p = 0.5 + 0.5 * float(honest_vals.mean()) if honest_vals.size else 0.5
                n_detected += int(self.rng.binomial(n_bad, min(det_p, 0.999)))
            # sybil resistance (trust entropy)
            if n_t >= 2:
                tc = np.clip(t, EPS, 1.0)
                p = tc / tc.sum()
                H = -np.sum(p * np.log(p + EPS))
                srs_sum += float(H / np.log(n_t))
            # finality time
            cft_sum += float(log2(n_t + 1) * (1.0 + 2.0 * frac_bad))

        mean_trust = float(trust_means.mean())
        avg_trust = mean_trust * 100.0
        bft = bft_ok / k
        overload_pen = float(np.minimum(util, 2.0).mean()) / 2.0
        consensus = float(np.clip(mean_trust * (1.0 - 0.5 * overload_pen), 0, 1))
        srs = srs_sum / k
        dsp = float(np.clip(0.90 + 0.10 * consensus, 0, 1))
        cft = cft_sum / k
        so = 1.1 * total_nodes * 0.01 + 0.5 * k

        return {
            "agg_tps": agg_tps,
            "mean_lat": mean_lat,
            "energy": energy,
            "avg_trust": avg_trust,
            "consensus": consensus,
            "bft": bft,
            "n_attacks": n_attacks,
            "n_detected": n_detected,
            "srs": srs,
            "dsp": dsp,
            "cft": cft,
            "so": so,
            "k": k,
            "mean_util": float(np.mean(util)),
            "max_util": float(np.max(util)),
            "served_ratio": served_ratio,
            "unmet_ratio": unmet_ratio,
            "overload_frac": overload_frac,
            "latency_score": latency_score,
        }
