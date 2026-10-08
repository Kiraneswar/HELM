# Solutions to Every Negative — Minimal and Optimal

For each weakness: a **minimal** fix (edit/reframe, no new experiment) and an
**optimal** fix (what actually removes the objection). Items marked **DONE NOW** I
executed in this session; results are real.

---

## Negative 1 — No accuracy improvement (HELM ties Random/Oort)

**Why it's there:** on these benchmarks, *which* clients you pick barely changes
accuracy; random is near-optimal.

**Minimal (DONE):** stop selling accuracy. Reframe the contribution as *equal accuracy
at lower cost and higher fairness*, and show it with the cost-to-accuracy plot
(accuracy vs cumulative energy) already added — HELM reaches any target accuracy at
lower energy than Centralized RL. "Same accuracy, less energy, fairer" is a valid,
defensible win, not a loss.

**Optimal (design change, offered):** make selection *count* a learned action
(variable budget) instead of a fixed per-cluster top-$k$. Then HELM can trade a small
amount of accuracy for a large energy saving and expose a controllable
accuracy--energy knob — turning "we tie on accuracy" into "we can hit any point on the
accuracy--energy curve the operator wants." This is the single highest-value upgrade
and also fixes Negative 2. Implementation: replace masked top-$k$ with per-client
Bernoulli selection (sigmoid over logits), update the PPO log-probability accordingly,
retrain. ~1 focused engineering session + re-runs. I can implement it on request.

---

## Negative 2 — The RL reward doesn't steer / RL contributes little

**Why it's there:** selection size is a fixed budget, so the energy-reward term acts
on a quantity the policy can't change (we proved weight 0 vs 5 → identical behavior).

**Minimal (DONE):** state it honestly (now in Limitations) and credit the *structure*,
not the reward. This is actually a clean scientific message: "you get fairness/energy
discipline from the hierarchy for free, without reward tuning."

**Optimal (same fix as N1):** variable-budget action makes the energy/fairness reward
*active* — the policy then genuinely optimizes the multi-objective trade-off, and the
reward-weight sweep becomes a real steerability result (a Pareto knob). This converts
the "RL does nothing" objection into "the RL learns the operator's chosen trade-off."

---

## Negative 3 — Scale and scope are thin  — **DONE NOW (largely resolved)**

**What I ran:** HELM vs Centralized RL on **real MNIST at 40 → 80 → 160 → 320
clients** (3 seeds each). Result — the thesis holds and strengthens with scale:

| N | HELM sel | Cent sel | HELM strag | Cent strag | HELM Gini | Cent Gini | acc (H vs C) |
|---|---|---|---|---|---|---|---|
| 40 | 10.5 | 14.4 | 7.0 | 11.0 | 0.406 | 0.482 | 80.4 vs 79.4 |
| 80 | 25.2 | 35.8 | 15.3 | 26.0 | 0.355 | 0.421 | 84.8 vs 84.7 |
| 160 | 47.0 | 65.4 | 29.2 | 47.7 | 0.377 | 0.450 | 86.3 vs 86.2 |
| 320 | 98.8 | 138.7 | 62.3 | 102.2 | 0.378 | 0.436 | 86.9 vs 86.9 |

HELM ties accuracy at every scale while the centralized selector's over-selection and
straggler count grow without bound, and the fairness gap persists. Added to the paper
as a 4-point scaling figure (`scaling_curve.png`). Scope is now: **two real datasets,
two architectures, 40–320 clients** — solid for a medium venue.

**Remaining (optional, optimal):** a churn sweep (0–50%) and a genuine cold-start
federation test would round out robustness. Both are cheap MLP runs; not yet done.

---

## Negative 4 — Related work incomplete (no recent citations) — **DONE NOW**

**What I did:** literature search; added five verifiable 2024–2025 references and a
positioning paragraph. The closest is *energy-aware MARL dual-selection* (2024) — I
position HELM as complementary (hierarchical decomposition + budget-controlled fairness
analysis + honest ablations), not a competitor claiming to beat it. Others: an
energy-efficient AIoT selector (2025), a DDPG secure-and-fair selector (2024), a
dynamic fairness-guarantee selector for heterogeneous edge (2024), and a 2024 client-
selection survey.

**Note:** I gave title/venue/year from search results; **verify each in a reference
manager and complete author lists before submitting** (same caveat as before — I don't
fabricate full author strings).
Sources:
- https://awesomepapers.io/federated-learning/papers/2405.08183
- https://link.springer.com/article/10.1186/s13638-025-02435-2
- https://onlinelibrary.wiley.com/doi/full/10.1155/2024/2314019
- https://jcst.ict.ac.cn/article/doi/10.1007/s11390-023-2972-9
- https://www.sciencedirect.com/science/article/abs/pii/S138912862400495X

---

## Bonus negatives from the strict review, and their fixes

**Energy model is modeled, not measured (Negative H).** Minimal (DONE): argued
invariance (all methods share the model, so rescaling any constant is a common factor;
the count-matched experiment shows energy is count-driven) and labeled it modeled.
Optimal: measure on one real device — future work.

**Single fairness metric (Negative I).** Minimal: keep Gini + the count-matched result
(strongest evidence). Optimal: add participation entropy / min-participation — needs
per-client selection logging (one-line instrumentation + re-run); not yet done.

**Reproducibility (Negative L).** DONE: added a PPO/FL hyperparameter + environment line
and released configs/logs/scripts.

---

## Where this leaves you now

- **Negatives 3 and 4: substantially resolved this session** (scaling to 320; recent
  related work added).
- **Negatives 1 and 2: honestly contained by reframing; fully removed only by the
  variable-budget redesign** — the one high-value change still open, which I can
  implement next.
- The paper (`HELM_paper_v3.pdf`) now carries: the equal-budget structural-fairness
  result, the 320-client scaling curve, cost-to-accuracy plots, recent related work,
  corrected claims, and honest limitations.

**Recommended order from here:** (1) verify the new citations; (2) decide on the
variable-budget redesign — it is what turns the two remaining negatives into positives
and materially raises novelty; (3) optional churn + entropy additions for polish.
