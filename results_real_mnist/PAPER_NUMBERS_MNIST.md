# FedRL-CS — Real-Data (MNIST) Results

**Setup.** Same pipeline as the synthetic experiments, with the synthetic
`make_classification` pool replaced by **real MNIST digits** (784-dim, standard
normalization). 40 clients / 4 clusters, Dirichlet non-IID (α=0.5), 100 FL
rounds, **5 seeds**. Each seed draws a different real subset (8,000 train images
partitioned across clients + 1,600 held-out test images), so the ± values are
genuine cross-seed variance on real data. RL selection, policy federation, the
grounded energy model, metrics, and analysis code are all unchanged.

## Main comparison (mean ± std over 5 seeds)

| Method | Final test acc (%) | Total energy (J) | Final Gini | Selected/rd | Stragglers/rd |
|---|---|---|---|---|---|
| **Federated MARL (Ours)** | **89.25 ± 1.21** | **513.2 ± 53.6** | **0.496 ± 0.016** | 8.00 | 4.93 |
| Random | 88.49 ± 1.02 | 515.5 ± 56.7 | 0.494 ± 0.018 | 8.01 | 4.98 |
| Oort (heuristic) | 88.76 ± 0.70 | 501.7 ± 44.4 | 0.509 ± 0.022 | 8.01 | 4.66 |
| Centralized RL | 88.61 ± 1.39 | 755.4 ± 87.4 | 0.581 ± 0.016 | 12.19 | 9.11 |
| Full participation | 76.92 ± 5.27 | 795.0 ± 94.3 | 0.578 ± 0.014 | 12.83 | 9.74 |

## Paired significance (ours − baseline, paired over seeds)

| vs baseline | Accuracy | Energy | Gini |
|---|---|---|---|
| Random | +0.76 pp (p=0.26, ns) | −0.5% (p=0.44, ns) | +0.5% (ns) |
| Oort | +0.49 pp (p=0.19, ns) | +2.3% (p=0.17, ns) | −2.4% (p=0.11, ns) |
| Centralized RL | +0.64 pp (ns) | **−32.1% (p=2.1e-4)** | **−14.6% (p=4.4e-7)** |
| Full participation | **+16.0% (p=5.8e-3)** | **−35.5% (p=1.9e-4)** | **−14.2% (p=1.8e-6)** |

## Honest interpretation (what the real data actually shows)

1. **No accuracy advantage over the good selection baselines.** On real MNIST,
   ours (89.3%) ties Random (88.5%), Oort (88.8%), and Centralized RL (88.6%) —
   all differences are within noise (p > 0.19). This *confirms* the earlier
   synthetic finding: the contribution is not higher accuracy. Ours does beat
   *Full participation* (+16 pp) because indiscriminately using every client
   floods aggregation with stragglers/noise and hurts accuracy.

2. **The energy and fairness advantages are real but specific.** They are large
   and highly significant **against Centralized RL (−32% energy, −15% Gini) and
   Full participation (−36% / −14%)**, and they come from a clear, verifiable
   mechanism: the centralized-RL selector *degenerates into over-selection*
   (~12 clients/round, ~9 stragglers), while ours stays lean (~8/round, ~5
   stragglers). **Federating the selection policy is what prevents that
   degeneration** — this is the structural result the paper should claim.

3. **Against the cheap baselines (Random, Oort) it is a tie**, not a win, on all
   three metrics. On mildly non-IID MNIST, selecting a small random/heuristic
   subset is already near-optimal, so there is little room to beat it. This must
   be stated plainly — the honest claim is *"matches the best lightweight
   baseline while learning to do so, and avoids the instability that sinks the
   centralized-RL alternative,"* not *"beats all baselines."*

**Bottom line for the paper.** The real-MNIST numbers support a narrower, honest
claim than the synthetic run implied: FedRL-CS attains the accuracy of the best
baselines at the *lowest* energy/fairness tier (matching Random/Oort, far better
than Centralized RL/Full), and policy federation is the demonstrable reason the
learned selector does not blow up its cost the way a centralized RL selector
does. The headline "−28% energy / −15% Gini vs baselines" from synthetic data
does **not** transfer to real MNIST as a blanket claim — it holds vs the RL/Full
baselines only. Report it that way.

Files: `mnist_comparison_table.csv`, `mnist_paired_stats.csv`,
`mnist_accuracy_overlay.png` (this directory).
