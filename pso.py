"""
pso.py  -  Adaptive Guided Best-Worst PSO (AGBWPSO).

Optimises a particle X = [k, mu_1..mu_kmax] to minimise a fitness function.
Worst-particle repulsion (c3 term) is the distinguishing feature.
"""
import numpy as np


class AGBWPSO:
    def __init__(self, rng, n_particles=8, iters=15, kmax=20,
                 w_max=0.9, w_min=0.35, c1=2.05, c2=2.05, c3=1.5):
        self.rng = rng
        self.n = n_particles
        self.iters = iters
        self.kmax = kmax
        self.w_max, self.w_min = w_max, w_min
        self.c1, self.c2, self.c3 = c1, c2, c3

    def optimise(self, fitness, demand_total):
        D = 1 + self.kmax  # k plus kmax capacities
        # init particles
        X = np.zeros((self.n, D))
        X[:, 0] = self.rng.integers(2, self.kmax + 1, self.n)
        X[:, 1:] = self.rng.uniform(1e2, 350.0, (self.n, D - 1))
        V = self.rng.normal(0, 1, (X.shape))

        P = X.copy()
        Pf = np.array([fitness(x) for x in X])
        g_idx = int(np.argmin(Pf))
        G = P[g_idx].copy()
        Gf = Pf[g_idx]
        history = [Gf]

        for it in range(self.iters):
            w = self.w_max - (self.w_max - self.w_min) * it / max(self.iters - 1, 1)
            w_idx = int(np.argmax(Pf))
            W = P[w_idx]                       # worst-known position
            r1 = self.rng.random((self.n, D))
            r2 = self.rng.random((self.n, D))
            r3 = self.rng.random((self.n, D))
            V = (w * V
                 + self.c1 * r1 * (P - X)
                 + self.c2 * r2 * (G - X)
                 - self.c3 * r3 * (W - X))     # repulsion from worst
            X = X + V
            # clip each dimension separately (NumPy >=1.25 safe)
            X[:, 0] = np.clip(np.round(X[:, 0]), 2, self.kmax)
            X[:, 1:] = np.clip(X[:, 1:], 1e2, 350.0)

            f = np.array([fitness(x) for x in X])
            improved = f < Pf
            P[improved] = X[improved]
            Pf[improved] = f[improved]
            g_idx = int(np.argmin(Pf))
            if Pf[g_idx] < Gf:
                Gf = Pf[g_idx]
                G = P[g_idx].copy()
            history.append(Gf)

        k_star = int(G[0])
        caps = G[1:1 + k_star]
        return k_star, caps, Gf, history
