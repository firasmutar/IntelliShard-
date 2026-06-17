"""
qlearn.py  -  Tabular epsilon-greedy Q-learning agent (enhanced).

Improvements over v1:
  * epsilon decay horizon is configurable (set to the real training length)
  * 7-dim state incl. unmet-demand ratio (agent can now see backlog)
  * action masking: illegal actions (merge below k_min, split above k_max)
    are never selected, so exploration isn't wasted
  * optimistic init (small positive) to encourage early exploration
"""
import numpy as np

ACTIONS = ["do_nothing", "split", "merge", "rebalance"]


class QAgent:
    def __init__(self, rng, alpha=0.15, gamma=0.92,
                 eps_start=1.0, eps_end=0.05, eps_decay_steps=20000,
                 k_min=4, k_max=20):
        self.rng = rng
        self.alpha = alpha
        self.gamma = gamma
        self.eps_start, self.eps_end = eps_start, eps_end
        self.eps_decay_steps = eps_decay_steps
        self.k_min, self.k_max = k_min, k_max
        self.Q = {}
        self.t = 0

    def epsilon(self):
        frac = min(1.0, self.t / self.eps_decay_steps)
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def _bins(self, val, edges):
        return int(np.digitize([val], edges)[0])

    def encode(self, feat):
        return (
            self._bins(feat["mean_util"], [0.4, 0.7, 0.9, 1.1]),
            self._bins(feat["max_util"], [0.7, 1.0, 1.5]),
            self._bins(feat["util_std"], [0.1, 0.25, 0.5]),
            self._bins(feat["trust"], [0.5, 0.75]),
            self._bins(feat["rho_c"], [0.6, 0.8]),
            self._bins(feat["k"], [7, 11, 15]),
            self._bins(feat.get("unmet", 0.0), [0.05, 0.2, 0.4]),  # backlog
        )

    def _q(self, s):
        if s not in self.Q:
            self.Q[s] = np.full(len(ACTIONS), 0.1)  # optimistic init
        return self.Q[s]

    def legal_mask(self, k):
        """Return boolean mask of legal actions given current shard count."""
        mask = np.ones(len(ACTIONS), dtype=bool)
        if k >= self.k_max:
            mask[1] = False   # cannot split
        if k <= self.k_min:
            mask[2] = False   # cannot merge
        return mask

    def act(self, s, k):
        mask = self.legal_mask(k)
        legal = np.where(mask)[0]
        if self.rng.random() < self.epsilon():
            return int(self.rng.choice(legal))
        q = self._q(s).copy()
        q[~mask] = -np.inf
        return int(np.argmax(q))

    def update(self, s, a, r, s2):
        q = self._q(s)
        q2 = self._q(s2)
        q[a] += self.alpha * (r + self.gamma * np.max(q2) - q[a])

    def step(self):
        self.t += 1
