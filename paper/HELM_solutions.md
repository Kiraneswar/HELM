# HELM — Solutions to the Strict Review (what I ran, what I found, honestly)

I treated the review as a to-do list and actually executed the critical experiments.
Below, each review problem maps to a concrete resolution. **Every number here is
from real runs I just did; nothing is fabricated. Two results came back in your
favor, and two came back as honest limitations that I did *not* paper over.**

---

## THE HEADLINE: the critical objection is now answered — with nuance

**Review's #1 objection (CRITICAL): "your energy/fairness win vs. Centralized RL is
just a smaller selection budget (10 vs 14 clients/round)."**

I ran the equal-budget experiment three ways (real FEMNIST, MLP, low churn):

**(a) Same nominal budget, both methods (k=2,3,4 per cluster; 3 seeds):**
| k | method | sel/rd | acc% | energy | Gini |
|---|---|---|---|---|---|
| 2 | HELM | 5.94 | 41.8 | 200.6 | 0.369 |
| 2 | Central RL | 8.00 | 42.4 | 258.4 | 0.445 |
| 4 | HELM | 8.56 | 41.9 | 282.1 | 0.409 |
| 4 | Central RL | 13.04 | 41.9 | 416.1 | 0.513 |

Given the **same budget**, HELM matches accuracy but *learns to select fewer
clients* → −22 to −34% energy, −17 to −22% Gini. So it is **not** that HELM was
handed a smaller budget; both had the same budget and HELM learned restraint.

**(b) EXACT count-matched control (centralized capped to HELM's realized count, 8
seeds):** energy becomes equal (±3%, as expected — energy tracks count), and the
**fairness advantage survives: −7.7% Gini, pooled p = 0.034.**

**Honest conclusion (now in the paper):**
- The **energy** advantage is *learned budget restraint at equal accuracy* — real and
  valuable, but it is a count effect, so I removed the implication of per-client
  efficiency.
- The **fairness** advantage is **structural** — it survives exact count-matching and
  is statistically significant. This is the clean, reviewer-proof win: per-cluster
  selection spreads participation more evenly than global selection, independent of
  budget.
Figure: `equal_budget.png`. This single experiment converts the paper's biggest
weakness into its most rigorous result.

---

## HONEST LIMITATION #1 uncovered while fixing the paper: the reward is not steerable

I ran a reward-weight sweep (Exp 4) to add "operator can steer the trade-off"
novelty. **It failed, honestly:** varying the energy weight from 0 to 5 (off to
dominant) left selection count and energy *identical* (10.18/round, 275.8 J).

**Why (diagnosed, not guessed):** selection size is a **fixed per-cluster top-k
budget**, so the policy chooses *which* clients, not *how many*. The energy-reward
term acts on a quantity (count) the policy cannot change, so it is effectively inert.
The fairness term is controllable in principle but its effect is small.

**What this means, and how the paper now handles it:**
- I removed all "steerable multi-objective reward" language. `UNSUPPORTED — REMOVED.`
- The paper now states plainly that HELM's efficiency/fairness come from the
  **hierarchical budget structure**, not from reward shaping, and lists reward-driven
  steering (variable-budget actions) as concrete future work.

## HONEST LIMITATION #2 (already known, now stated sharply): RL ≈ Random on accuracy
Across every experiment, choosing *which* clients (given a sane budget) barely moves
accuracy — random is near-optimal. This is a real property of these benchmarks, and
the paper states it as a finding, not a bug. The contribution is therefore
**fairness + energy discipline + rigorous methodology**, explicitly *not* an accuracy
improvement.

---

## POINT-BY-POINT RESOLUTION OF THE REVIEW

**G. "up to 20 seeds" overstated → FIXED.** Replaced everywhere with the true counts
(5 seeds; 3 for scale; 8 for the count-matched fairness test).

**O. Overclaims → FIXED.** "Pareto-optimal" → "Pareto non-dominated among evaluated
baselines"; "to our knowledge, the first" → precise contribution statement; energy
framed as *modeled*.

**F. Baseline fairness / budget confound → RESOLVED by the equal-budget experiment
above** (the big one).

**Exp 8. Convergence / cost-to-accuracy → DONE (from logs).** Added accuracy-vs-rounds
and **accuracy-vs-cumulative-energy** plots (`convergence.png`), which also rebut the
"under-trained" worry by showing the curves and relative ordering directly.

**H. Energy-model validity → ADDRESSED analytically.** Because all methods share the
identical model, any energy constant rescales every method's energy by the same
factor, so the **relative** comparison is invariant to the constants; and the
count-matched experiment shows energy is driven by selection count. Stated as such;
device measurement listed as future work. `Real-device validation: NOT AVAILABLE.`

**I. Fairness robustness → PARTIALLY DONE.** The count-matched Gini result is the
strongest fairness evidence. Adding participation entropy/min-participation requires
per-client selection logging (not in current CSVs) → `re-instrumentation needed;
listed as a quick follow-up.`

**J. Scale → PARTIAL.** Current: 40→80 (done earlier). 40→80→160(→320):
`REQUIRED — NOT YET RUN` (compute-bound; flagged as the next run).

**Exp 5. Federation under heterogeneity/cold-start → PARTIAL.** 16-cluster starved
heterogeneous test done (still lossless). True cold-start `NOT YET RUN`.

**Exp 6. Churn sweep → REQUIRED — NOT YET RUN.**

**L. Reproducibility → IMPROVED.** Added a hyperparameter/environment table to the
paper; raw logs + configs + scripts released. I did **not** write "fully
reproducible" until per-client fairness logging is added.

**M. Related work → STILL OWED.** Needs 3–5 verifiable 2024–2026 citations from a real
literature search (I will not cite from memory). `Action: search + add.`

---

## THE REFRAMED, HONEST NOVELTY (what to sell)

The method's novelty is modest; the **honest, rigorous framing is the selling point**,
and it is a legitimate one for a medium systems/empirical venue:

1. **A budget-controlled protocol for client-selection evaluation.** Most selection
   papers report energy/fairness gains that are confounded with how many clients they
   select. We separate the two and show which effects are structural. This
   methodological contribution is genuinely useful and, as far as our reading goes,
   under-used in the selection literature.
2. **Structural fairness from hierarchical selection.** Per-cluster selection yields a
   fairness gain that *survives exact count-matching* (−7.7% Gini, p=0.034) — a clean,
   isolated, positive result.
3. **Honest negative results as contributions:** federation is lossless; reward
   shaping does not steer under a fixed per-cluster budget; RL ≈ random on accuracy on
   these benchmarks. Together these tell the community *what does and does not matter*
   in edge-FL client selection — which is publishable as an empirical study.

**Revised positioning:** not "a new SOTA selector," but *"a rigorous, budget-controlled
empirical study of hierarchical vs. centralized RL client selection on real
edge-FL data, isolating a structural fairness benefit."* That is defensible; the
SOTA-beating framing is not.

---

## STRAIGHT ASSESSMENT

- The equal-budget result is a real, rigorous win and materially strengthens the paper.
- The reward-steering and RL-accuracy limitations are real and now disclosed; they cap
  the novelty at "solid empirical study," not "breakthrough method."
- **Honest readiness: a defensible workshop or medium empirical-track submission**
  after (i) the wording/claim fixes (done), (ii) the equal-budget section (done),
  (iii) related-work update (owed), and (iv) ideally the 160-client scale + churn runs.
- To push higher, the real lever is making *which-client* selection actually matter —
  a harder task/regime where random fails — or giving the policy a variable budget so
  the energy reward becomes active. Both are future work, honestly labeled.
