"""
workload.py  -  Ethereum-calibrated streaming workload generator.

Produces one "window" per call. A window summarises 10 consecutive Ethereum
blocks (~120 s). All randomness is seeded for full reproducibility.

Calibration choices and their justification:
  * mean tx per window ~ Poisson, base 190, modulated by:
       - time-of-day factor peak(h) = 1 + 0.7 sin((h-8) pi/12)
       - weekend factor weekly(d) = 0.85 on Sat/Sun
       - surge factor 2.5x with probability 0.01 (MEV / airdrop bursts)
  * per-shard load split ~ Dirichlet(alpha=0.8)  -> heavy-tailed, matches
    the power-law account-activity distribution reported for Ethereum.
  * cross-shard ratio rho_c = 1 - 0.40/k  (more shards -> more cross-shard tx)
"""
import numpy as np


class EthereumWorkload:
    def __init__(self, seed=42, base_tx=2600):
        self.rng = np.random.default_rng(seed)
        self.base_tx = base_tx
        self.t = 0  # window index

    def _peak(self, hour):
        return 1.0 + 0.7 * np.sin((hour - 8) * np.pi / 12.0)

    def _weekly(self, day):
        return 0.85 if day in (5, 6) else 1.0

    def next_window(self, k):
        """Generate the next window for a topology of k shards.

        Returns a dict describing demand only (no strategy logic here).
        """
        # simulated wall-clock: 120 s per window
        hour = (self.t * 120 // 3600) % 24
        day = (self.t * 120 // 86400) % 7

        peak = max(0.2, self._peak(hour))
        weekly = self._weekly(day)
        surge = 2.5 if self.rng.random() < 0.01 else 1.0
        is_surge = surge > 1.0

        mean = self.base_tx * peak * weekly * surge
        n_tx = int(self.rng.poisson(mean))

        # split demand across shards with a heavy tail
        if k < 1:
            k = 1
        weights = self.rng.dirichlet(np.full(k, 0.8))
        shard_tx = (weights * n_tx).astype(float)

        rho_c = min(0.95, 1.0 - 0.40 / max(k, 1))
        gas_price = float(np.exp(self.rng.normal(0.0, 0.3)) * 30.0)  # gwei-ish

        self.t += 1
        return {
            "t": self.t - 1,
            "hour": hour,
            "day": day,
            "n_tx": n_tx,
            "shard_tx": shard_tx,        # length-k vector of tx demand
            "cross_shard_ratio": rho_c,
            "gas_price": gas_price,
            "is_surge": is_surge,
        }


if __name__ == "__main__":
    wl = EthereumWorkload(seed=42)
    tot = 0
    surges = 0
    for _ in range(2000):
        w = wl.next_window(10)
        tot += w["n_tx"]
        surges += w["is_surge"]
    print(f"2000 windows: mean tx/window = {tot/2000:.1f}, surges = {surges}")
