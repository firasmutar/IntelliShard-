# IntelliShard++

Reference implementation and reproducibility package for the paper:

> **IntelliShard++: A Hybrid Reinforcement Learning and Particle Swarm Optimisation Framework for Dynamic Blockchain Sharding**
> Firas H. Neamah, Sajjad H. Hasan, Hussein Akeel Hussein Alaasam, Hussein Alrammahi, Yahya Mahdi Hadi Al-Mayali.
> *International Journal of Intelligent Engineering and Systems (IJIES)*.

IntelliShard++ is a cost-aware dynamic blockchain-sharding controller. A tabular ε-greedy
**Q-learning** agent decides *when* to act (do nothing, split, merge, or rebalance), and an
**Adaptive Guided Best-Worst Particle Swarm Optimiser (AGBWPSO)** decides *how* to act
(the number of shards and per-shard capacities) when a rebalance is triggered. A
reconfiguration-cost term in the shared objective discourages needless restructuring.

This repository contains the **complete discrete-event simulator** and every script used to
produce the numbers and figures in the paper. There are no hidden or hard-coded results: every
value is computed by the code here.

---

## Headline results (N = 100,000 windows, seed 42)

| Strategy | Aggregate TPS | Mean latency | Avg. shards |
|---|---:|---:|---:|
| Baseline-10 (fixed 10 shards) | 1,571 | 47.94 | 10.0 |
| Baseline-20 (fixed 20 shards) | 2,084 | 14.98 | 20.0 |
| Reactive controller (ablation) | 1,524 | 188.47 | 2.0 |
| **IntelliShard++** | **2,290** | **13.13** | **11.6** |

IntelliShard++ delivers **+45.7 %** throughput over the conventional ten-shard baseline and
**+9.9 %** over a strongly provisioned twenty-shard baseline while using ~42 % fewer shards,
with the lowest mean latency of any strategy. These gains carry a small, **explicitly reported**
cost in Byzantine-fault-tolerance margin and worst-case latency.

Across **10 random seeds** the throughput gain is **+46.1 % ± 3.4 %** vs. Baseline-10
(paired t-test, p < 0.001) and **+10.1 % ± 2.5 %** vs. Baseline-20 (p < 0.001).

---

## Repository layout

```
IntelliShard-plus-plus/
├── workload.py          # Ethereum-calibrated workload generator (Poisson + Dirichlet)
├── shard_model.py       # Shard system: split / merge / set-capacity, metrics, queueing latency
├── pso.py               # AGBWPSO optimiser (8 particles, 15 iters, worst-particle repulsion)
├── qlearn.py            # Tabular epsilon-greedy Q-learning agent (7-dim state, action masking)
├── run_sim.py           # Runs all 4 strategies on the same workload; writes results.json
│
├── multiseed.py         # 10-seed variability study   -> multiseed_results.json
├── synergy.py           # RL-only / PSO-only ablation  -> q2_synergy.json
├── sweeps.py            # Shard-count + cost-weight sweeps -> q3_*.json, q4_*.json
├── make_figures.py      # Paper figures from results.json
├── make_r3_figures.py   # Statistical / synergy figures
│
├── results/ *.json      # Pre-computed result files used in the paper (at repo root)
├── figures/ *.png       # Pre-rendered figures (300 DPI)
├── requirements.txt
└── LICENSE
```

---

## Quick start

```bash
git clone https://github.com/<author-account>/IntelliShard-plus-plus.git
cd IntelliShard-plus-plus
pip install -r requirements.txt
```

### Reproduce the main result

```bash
python run_sim.py 100000        # ~3-5 min; writes results.json and prints the comparison table
python make_figures.py          # regenerates the main paper figures into figures/
```

A smaller run for a quick check:

```bash
python run_sim.py 5000
```

### Reproduce the statistical and component analyses (paper Section 8)

```bash
python multiseed.py 600         # 10 seeds x 20,000 windows (resumable; budget in seconds)
python synergy.py               # RL-only vs PSO-only vs full   (Q2 synergy)
python sweeps.py                # shard-count sweep (Q3) + cost-weight sweep (Q4)
python make_r3_figures.py       # variability, reward-correlation, synergy, sweep figures
```

`multiseed.py` takes a time budget in seconds and checkpoints after every seed, so it can be
re-run until all ten seeds are complete.

---

## How the simulator works

**Workload (`workload.py`).** Transaction arrivals follow a Poisson process whose mean is
modulated by a time-of-day term, a weekday/weekend factor, and an occasional surge. Per-shard
demand follows a Dirichlet distribution that reproduces the heavy-tailed account activity seen
in Ethereum. The generator is calibrated to public Ethereum block statistics.

**Shard model (`shard_model.py`).** Maintains the set of shards, supports `split`, `merge`, and
`set_capacity`, caps per-shard capacity (so high demand genuinely needs more shards), and each
window computes throughput, queueing latency, and six security metrics (consensus score, BFT
ratio, attack-detection rate, Sybil-resistance score, double-spend prevention, finality time).

**Q-learning (`qlearn.py`).** A 7-dimensional discretised state (utilisation mean / max / spread,
mean trust, cross-shard ratio, shard count, unmet-demand ratio). Illegal actions are masked.
Exploration decays over the first 60 % of the run, then the policy is exploited. Learning rate
α = 0.15, discount γ = 0.92.

**AGBWPSO (`pso.py`).** Eight particles, fifteen iterations, inertia 0.9 → 0.35, acceleration
coefficients c1 = c2 = 2.05 (constriction), plus a worst-particle repulsion term c3 = 1.5.
Per-shard capacities are clipped to [100, 350] TPS.

**Strategies (`run_sim.py`).** Four are run on the *same* workload and seed: a fixed ten-shard
baseline, a well-provisioned twenty-shard baseline, a reactive threshold controller (split above
80 % / merge below 20 %, no cost and no cooldown — an **ablation**, not a reproduction of any
specific published protocol), and IntelliShard++.

---

## Key hyperparameters

| Parameter | Value | Parameter | Value |
|---|---|---|---|
| Windows N | 100,000 | PSO particles | 8 |
| Seed | 42 | PSO iterations | 15 |
| RL learning rate α | 0.15 | PSO inertia ω | 0.9 → 0.35 |
| RL discount γ | 0.92 | PSO c1 = c2 | 2.05 |
| ε start / end | 1.0 / 0.05 | PSO c3 (repulsion) | 1.5 |
| ε horizon | 60 % of run | shard count k range | 4 – 20 |
| Capacity cap / shard | 350 TPS | malicious-node ratio | 8 % |

All values are defined in `run_sim.py` (`WEIGHTS`, `RECONF_COST`) and the agent / optimiser
constructors, and can be changed there.

---

## Reproducibility notes

* All randomness is seeded; `run_sim.py 100000` is deterministic and reproduces `results.json`.
* The pre-computed `*.json` result files and `figures/*.png` in this repository are exactly those
  used in the manuscript, so the figures can be regenerated without re-running the long jobs.
* The reactive baseline is intentionally a simplified threshold controller used as an ablation;
  it is **not** a reimplementation of DynaShard or any other named protocol, and the paper says so.

## Citation

If you use this code, please cite the IJIES paper above. A BibTeX entry will be added once the
final volume / issue / page numbers are assigned.

## License

Released under the MIT License (see `LICENSE`).
