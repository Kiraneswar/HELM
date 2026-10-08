# NOTES.md — Honest Assessment of Results

These notes represent my (the author/implementer's) direct read of the results,
including observations that are surprising, inconsistent, or not yet resolved.
Nothing has been smoothed or omitted.

---

## Did Federated MARL beat Centralized RL?

**Marginally, yes — but not decisively.**

  Federated MARL (Ours): 64.0% ± 2.0%
  Centralized RL:        63.5% (1 seed)
  Random selection:      64.1% (1 seed)

The gap over Centralized RL is +0.5 percentage points.
This is within 1 std of federated_marl's own variance across seeds (2.0%).
**This is not a statistically significant result yet.** Centralized RL and all
other baselines were only run with 1 seed. You cannot make a clean claim of
superiority until all baselines are also run with num_seeds=5.

The energy advantage of federated_marl over centralized_rl is more meaningful:
  Total energy saved vs. CRL: 1.56 units
  (~7% more efficient)
  Gini improvement vs. CRL: 0.086
  (federated_marl is meaningfully fairer in participation)

The fairness improvement is real and consistent across all 5 seeds. Federated
MARL's final Gini (0.496) is lower than all baselines,
suggesting the multi-cluster RL policies naturally spread participation more
evenly than centralized approaches.

---

## When Did Federation Help or Hurt?

**Federation helps in the middle of training (rounds 20-60)** where the accuracy
curves show federated_marl pulling ahead of random and oort. The likely mechanism:
  - Per-cluster policies that have learned something useful in some clusters
    transfer that knowledge to others via FedAvg on policy weights.
  - This is the core novelty claim and it appears to have some empirical support.

**Federation does not help in late rounds (70-100)** where all methods plateau
at similar accuracy levels. The plateau is driven by battery churn reducing the
client pool, not by the selection policy — so policy quality matters less.

**Alpha sweep**: If ablation ran, check ablation_results.csv. The expectation is
alpha=0.5 (soft update) should outperform alpha=0.0 (hard replace) and alpha=1.0
(no federation). If the curve is not unimodal around 0.3-0.5, this is a red flag.

**T_RL cadence sweep**: T_RL=1 (federate every round) should overfit to the
global average and hurt per-cluster specialization. T_RL=10 may not federate
often enough for knowledge transfer. T_RL=5 was the default and should be near
the sweet spot if the mechanism is real.

---

## Suspicious / Concerning Observations

### 1. Oort underperforms Random at round 100
  Oort:   61.8% (energy: 19.77)
  Random: 64.1% (energy: 19.80)

This is surprising. In the FL literature (Lai et al., 2021), Oort consistently
outperforms random selection, especially on non-IID data. Possible explanations:
  a) Oort's utility score prioritizes high-loss clients, which tend to have
     fewer local samples or more extreme label skew. These clients also deplete
     their batteries faster under battery drain, leaving Oort with a worse
     selection pool in later rounds compared to the uniform random baseline.
  b) The synthetic tabular dataset (make_classification) may not produce the
     kind of loss landscape where Oort's statistical utility score is informative.
     Oort was designed for MNIST/CIFAR where high local loss correlates with
     informativeness. That correlation may not hold here.
  c) Implementation error: worth re-checking the OortSelector's utility update
     and client-ranking logic before reporting these results.

**FLAG: Do not present Oort's performance as a clean finding. Either explain
the battery mechanism rigorously or re-run with battery_drain=0 to isolate it.**

### 2. Policy similarity metric appears uninformative
The policy_similarity curve in the RL diagnostics plot is near-zero and flat.
This is suspicious because after federated averaging, cluster policies should be
more similar to each other, not less. Possible causes:
  a) The cosine similarity is being computed on pre-federation weights (before the
     FedAvg step), not post. Check runner.py's order of operations.
  b) With 4 clusters and small policy networks, the pairwise cosine similarity may
     be dominated by the random initialization (all clusters start similar, then
     diverge, then re-converge at each T_RL step). If the metric is computed as
     "similarity vs. first cluster" rather than "mean pairwise similarity", the
     divergence phase would show low values even when federation is working.
  c) The reward signal is too weak to drive meaningfully different per-cluster
     policies. If all clusters are learning roughly the same thing, federation
     trivially succeeds but provides no real benefit.

**FLAG: Fix or validate the policy_similarity metric before claiming federation
improves cross-cluster coordination. Currently this metric does not support the
paper's core claim.**

### 3. All methods converge to similar final accuracy
At round 100, all methods are within 1.1 percentage points
of each other (62.9% to 64.1%). With only 1 seed for baselines,
it is impossible to say whether these differences are real or noise.

The energy and fairness differences are larger and more reliable as secondary
claims. If the paper's primary claim is accuracy superiority, the current results
are too noisy to support it cleanly.

---

## What Is Publication-Ready

- The system architecture (two-cadence FL+RL training loop) is implemented
  correctly and runs end-to-end.
- The energy efficiency and fairness improvements are consistent and directionally
  correct across all 5 seeds.
- The FL core sanity checks all pass: local overfitting works, FedAvg is correct,
  test set is properly held out.
- The ablation configs are defined and can produce systematic results.

## What Still Needs Work

1. Run all baselines with num_seeds=5 (currently only federated_marl has 5 seeds).
2. Fix or validate the policy_similarity metric.
3. Investigate and explain Oort's underperformance vs. random.
4. Consider re-running with battery_drain=0 as a control to isolate selection
   policy effects from churn effects.
5. If accuracy claim is primary: need larger dataset or more informative task to
   get a cleaner separation between methods.
