# HELM — Strict Reviewer Report, Editor Notes, and Action Plan

**Role:** hostile IEEE/ACM reviewer + advisor + editor, simultaneously.
**Ground rule I followed:** nothing invented. Where evidence is missing I write
`REQUIRED EXPERIMENT — NOT CURRENTLY AVAILABLE`; where a claim outruns the data I
write `UNSUPPORTED CLAIM — REMOVE OR VERIFY`. Negative results are kept.

**One-line verdict:** *PROMISING BUT MAJOR REVISION REQUIRED.* The core story
(hierarchical selection avoids centralized over-selection; Pareto trade-off on real
data) is real and honestly told, but two issues are currently **fatal to
acceptance as written**: (1) the seed-count wording overstates the statistics, and
(2) the central causal claim ("hierarchy, not budget") is not yet isolated by an
equal-budget experiment. Both are fixable — one by editing, one by one run.

---

## PART 1 — PUBLICATION-GRADE REVIEW (by heading, with severity)

**A. Novelty — MODERATE.**
The combination (per-cluster RL selection + grounded joule reward + Gini fairness +
an honest federation ablation) is a reasonable, not landmark, contribution. Each
ingredient exists separately (RL selection: Favor/Oort-style; fairness: q-FFL;
energy-aware FL). *Reviewer attack:* "incremental combination." *Defense that
works:* the multi-objective grounded framing + the honest isolation of hierarchy
vs. federation is the contribution; do not claim more.

**B. Technical contribution — MODERATE.** Solid systems/ML engineering, clearly
described. Not theoretically deep (no convergence/regret analysis). Acceptable at a
medium venue; a top venue would want theory or a much larger empirical study.

**C. Problem formulation — MINOR.** Two-tier model, grounded energy, Gini, deadline
are clearly defined. Missing: formal statement of the per-cluster budget `k` as a
constraint, and an explicit statement that total budget = `C·k` — which is exactly
the quantity a reviewer will interrogate (see F).

**D. Algorithmic contribution — MODERATE.** Algorithm 1 is clear. Missing details a
re-implementer needs: PPO clip range, GAE λ, actor/critic architecture and widths,
entropy coefficient, optimizer/LR, how logits map to a top-k *sample* (with or
without replacement), and how the critic is used. `MAJOR` for reproducibility if
left as is.

**E. Experimental validity — MAJOR.** Real data (FEMNIST by writer, MNIST), two
architectures, six baselines, significance testing — genuinely good for a UROP-scale
paper. But absolute accuracy is low (FEMNIST CNN ~45%, still climbing at the round
budget) and the evaluation is academic-scale. *Reviewer attack:* "under-trained
models; can we trust the ranking?" *Partial defense:* the ranking is stable across
MLP/CNN and two datasets; state this explicitly and show a convergence curve
(Experiment 8).

**F. Baseline fairness — CRITICAL.** Centralized RL selects **~14 clients/round**;
HELM selects **~10**. The paper's headline energy/fairness gains vs. Centralized RL
are therefore **confounded with selection budget**. A skeptical reviewer will say
the win is trivially "we selected fewer clients." **This is the most important
weakness in the paper.** → `REQUIRED EXPERIMENT — Equal-budget Centralized RL`
(Experiment 1). Without it, the central claim is not established.

**G. Statistical validity — CRITICAL (but easy to fix).** The abstract and setup
say **"up to 20 seeds."** Every real-data table uses **5 seeds** (scale uses **3**).
No reported real-data result uses 20. `UNSUPPORTED CLAIM — REMOVE OR VERIFY`:
change all "up to 20 seeds" to the true per-table counts ("5 seeds; 3 for the scale
study"). Reporting mean±std and paired tests is appropriate; with n=3–5, also report
that n is small and prefer exact p-values (already done) — but **do not** imply more
seeds than were run. This one wording issue alone can trigger a desk-level trust
problem; fix it everywhere.

**H. Energy-model validity — MAJOR.** Energy is in joules but uses order-of-magnitude
constants with **no device measurement**. The paper already discloses this, which is
good. *Reviewer attack (systems reviewer):* "these numbers are not physical." →
`REQUIRED: sensitivity analysis over (e_mac, e_bit, φ, η)` (a cheap re-computation
from existing logs, no new training) + soften language to "modeled energy, relative
comparison valid." Do not present as measured energy anywhere.

**I. Fairness evaluation — MODERATE.** Single metric (Gini). Adding **one**
complementary metric strengthens the claim: report **participation entropy** or
**min-participation** (both computable from existing selection logs, no new runs).
Jain's index is monotone-ish with what Gini already shows, so it adds least; entropy
+ worst-case participation add most. Do this from logs you already have.

**J. Scalability — MAJOR.** 40→80 is only two points and 80 uses just 3 seeds. A
single doubling is a weak scaling claim. → `REQUIRED: 40→80→160(→320) if compute
allows`, same metrics. The trend (stragglers 9.5→25.7 for Centralized RL) is
compelling *if* extended; as-is it's suggestive.

**K. Real-world relevance — MODERATE.** Simulated links/battery. The paper mostly
distinguishes simulation from reality, but the visual-abstract results banner reads
as if physical. Ensure every energy/straggler number is labeled "simulated/modeled."

**L. Reproducibility — MAJOR.** Claims logs/pipeline are released. Missing an explicit
artifact table (versions, hyperparameters, seeds, exact configs). Provide it (Part 3)
and do not write "fully reproducible" until the repo actually contains eval + plotting
scripts and raw logs.

**M. Related work — MAJOR.** Only 7 references; **no 2024–2026** client-selection,
energy-aware, or hierarchical-FL work. A reviewer will immediately name missing
recent methods. → add a recent-work paragraph (Part 5). This is a common desk-reject
trigger for "insufficient positioning."

**N. Writing/organization — MINOR (a strength).** The FROST-style narrative is clear
and the honesty is a plus. Keep it.

**O. Claims too strong — MAJOR.**
- "Pareto-optimal" (visual abstract) → **"Pareto non-dominated among the evaluated
  baselines."** `UNSUPPORTED CLAIM`.
- "to our knowledge, the first to…" (related work) → replace with a precise
  contribution statement; you have not run a systematic priority search.
  `UNSUPPORTED CLAIM`.
- "up to 20 seeds" → see G.

**P. Missing experiments — see Part 2.** Equal-budget (critical), sensitivity, larger
scale, heterogeneity+cold-start, churn sweep, convergence/energy-to-target.

**Q. Likely reviewer objections — see Part 6.**

---

## PART 2 — MISSING EXPERIMENTS, PRIORITIZED

Each marked by whether it needs new training or only re-analysis of existing logs.

**EXP 1 — Equal-budget Centralized RL. `REQUIRED — NOT AVAILABLE`. (New runs, small.)**
*Why it is necessary:* it is the only way to show the advantage comes from
*hierarchy* and not from HELM simply selecting fewer clients (currently 10 vs 14).
Constrain Centralized RL to the same total budget `C·k` per round (or constrain HELM
up to Centralized RL's budget), and compare accuracy, energy, Gini, stragglers,
wall-clock, and time-to-target-accuracy. **If HELM still wins at equal budget → the
hierarchy claim is established and the paper is much stronger. If the gap shrinks to
zero → the honest finding is "the benefit is a disciplined budget, which the
hierarchy enforces automatically," which is still publishable but must be stated.**
Do not predict the outcome; run it. This is the single highest-value experiment.

**EXP 2 — Energy sensitivity / (optional) device validation. `REQUIRED (sensitivity)`.
(Re-analysis of logs; no new training.)**
Recompute total energy while sweeping each constant (e_mac, e_bit, φ, η) over ±1–2
orders of magnitude and show the *method ranking is invariant*. Real-device
measurement is `NOT AVAILABLE` and should be listed as future work, not claimed.

**EXP 3 — Deeper scaling 40→80→160(→320). `REQUIRED — NOT AVAILABLE`. (New runs; the
big ones may be compute-bound — use MLP, fewer seeds, state so.)**
Test whether the over-selection gap keeps widening. Even 40→80→160 at 3 seeds would
substantially strengthen J.

**EXP 4 — Reward-weight sensitivity. `REQUIRED — NOT AVAILABLE`. (New runs, cheap, MLP.)**
Sweep (w_a, w_e, w_f) over a small grid; show HELM stays on the frontier and how the
frontier moves. Directly answers "did you cherry-pick 1/0.1/0.05?"

**EXP 5 — Strong cluster heterogeneity + cold-start, federation ON/OFF. `PARTIAL`.**
Already done: 16 small heterogeneous clusters, ON vs OFF (still lossless, p>0.08).
`NOT AVAILABLE`: a genuine **cold-start** test (a new cluster joins late with a fresh
policy; does federation warm-start it?). This is the one setting where federation
*might* finally help; run it, report honestly either way.

**EXP 6 — Churn sweep 0/10/20/30/50%. `REQUIRED — NOT AVAILABLE`. (New runs, MLP, cheap.)**
Only mild churn tested. A churn sweep shows robustness and recovery.

**EXP 7 — Complementary fairness metric. `AVAILABLE FROM LOGS`. (Re-analysis only.)**
Add participation entropy and/or min-participation from existing selection logs.

**EXP 8 — Convergence / cost-to-accuracy. `AVAILABLE FROM LOGS`. (Re-analysis only.)**
Plot accuracy vs. rounds, accuracy vs. cumulative energy, and energy-to-reach-target.
You already log per-round accuracy and energy — this is a plotting task and it
directly rebuts "under-trained/unfair-cost" objections.

---

## PART 3 — CLAIMS TO SOFTEN (table)

| Current claim | Problem | Recommended claim |
|---|---|---|
| "up to 20 seeds" (abstract, §V, §I) | No real-data table uses 20; max is 5 (3 for scale) | "5 random seeds (3 for the 80-client scale study)" |
| "Pareto-optimal" (visual abstract) | Optimality not proven; only vs. evaluated baselines | "Pareto non-dominated among the evaluated baselines" |
| "to our knowledge, the first to…" | No systematic priority search | "We combine, in one selector, hierarchical per-cluster RL with a grounded multi-objective reward, and quantify the effect of federating the selection policy." |
| "cutting energy 26–33% … vs centralized RL" (headline) | Confounded with selection budget (10 vs 14) | Keep, but add: "at equal per-round selection budget, see Table [EXP1]" — only after running EXP 1 |
| Energy in joules (implies physical) | Modeled constants, no device data | "modeled energy (joules); constants shared across methods, so relative comparisons hold" |

---

## PART 4 — REPRODUCIBILITY CHECKLIST (fill before claiming "reproducible")

Present ✓ / Missing ✗ (from this project's actual state):
- ✓ Datasets + partitioning (FEMNIST by writer; MNIST Dirichlet α=0.5) — document exact writer IDs / seed.
- ✓ Cluster assignment, k, deadline, churn config (in YAML) — include the YAMLs in the repo.
- ✓ Reward weights, energy constants — in config.
- ✗ PPO hyperparameters in the paper (clip, GAE λ, entropy coef, LR, epochs, network widths) — **add to a table**.
- ✓ Rounds, seeds — but must be stated truthfully per table.
- ✓ Raw per-round CSV logs — release them.
- ✓ Plotting/analysis scripts — release them.
- ✗ Environment (Python/torch versions, hardware, wall-clock) — **add**.
Do not write "fully reproducible" until PPO hyperparameters + environment are in the
paper/repo.

---

## PART 5 — RELATED WORK TO ADD (positioning, not padding)

The paper has 7 refs and nothing from 2024–2026. Add a short paragraph positioning
against recent **client-selection / energy- or fairness-aware / hierarchical FL**
work. `NOTE:` I did not run a live literature search in this pass, so **do not cite
specific 2024–2026 titles from memory** — search yourself (or ask me to, with web
access) and add 3–5 genuinely relevant, verifiable ones. Candidate *areas* to cover:
recent RL/bandit client selection beyond Oort/PoC; energy- or carbon-aware FL
selection; hierarchical/clustered FL; multi-objective FL. For any recent method you
cannot implement fairly, position it in prose and say why a numerical comparison
would be unfair (different assumptions/objective), rather than reporting an unfair
number.

---

## PART 6 — THREE-REVIEWER ATTACK TEST

**Reviewer A (FL/ML).**
- "Novel?" → Combination-novel; softened claims required (Part 3). *Addressed after edits.*
- "Baselines fair?" → **Not yet** — budget confound. *Needs EXP 1.*
- "RL justified vs. a bandit?" → Add one sentence justifying sequential/stateful
  formulation (staleness, battery dynamics evolve). *Edit.*
- "Statistically convincing?" → Only after the seed wording is fixed and n stated
  honestly; consider adding seeds for the headline table if cheap. *Edit + optional runs.*
- "Accuracy competitive?" → Low absolute; rebut with convergence curve (EXP 8) and
  the cross-architecture stability argument. *Re-analysis.*

**Reviewer B (Edge/Systems).**
- "Energy realistic?" → **Not measured.** *Needs EXP 2 sensitivity + honest framing.*
- "Represents edge?" → Simulated; label clearly. *Edit.*
- "Scalability shown?" → Weak (2 points). *Needs EXP 3.*
- "Heterogeneity/churn/deadlines?" → Deadlines ✓, heterogeneity partial, churn ✗.
  *Needs EXP 5 cold-start + EXP 6.*
- "PPO overhead?" → Not reported. *Add wall-clock + per-round selection overhead
  (from logs / one timed run).*

**Reviewer C (skeptic).**
- "Gains just from smaller budget?" → **Currently unanswerable.** *EXP 1 is mandatory.*
- "Claims overstated?" → Yes in three places. *Part 3 edits.*
- "Datasets sufficient?" → Two is thin but acceptable for a medium venue with the
  scale + sensitivity additions.
- "Reproducible?" → Not until Part 4 gaps filled.
- "Does federation contribute?" → Honestly, no (in tested regimes); keep as a
  *lossless/free* framing + the cold-start test (EXP 5).
- "Generalize?" → Scope claims to cross-device image FL; don't over-generalize.

---

## PART 7 — FINAL ACCEPTANCE-READINESS REPORT

**A. Top 10 changes before submission**
1. Fix "up to 20 seeds" → true counts, everywhere. *(edit, critical)*
2. Run **EXP 1 equal-budget Centralized RL**; report it. *(run, critical)*
3. Soften "Pareto-optimal" → "non-dominated among evaluated baselines." *(edit)*
4. Remove "first to our knowledge"; state precise contribution. *(edit)*
5. Add PPO hyperparameter + environment table. *(edit)*
6. Add energy **sensitivity** analysis; reframe energy as modeled. *(re-analysis)*
7. Add convergence + energy-to-target plots (EXP 8). *(re-analysis)*
8. Add one complementary fairness metric (entropy/min). *(re-analysis)*
9. Add 3–5 verifiable 2024–2026 related-work citations. *(literature search)*
10. Label all energy/straggler numbers "simulated/modeled" incl. the visual abstract.

**B. Top 5 experiments by priority**
1. Equal-budget Centralized RL *(critical)* → 2. Reward-weight sensitivity → 3.
Deeper scaling 40→80→160 → 4. Churn sweep → 5. Federation cold-start.

**C. Claims that must be softened:** 20-seed, Pareto-optimal, "first," physical-energy
implication, unrestricted "beats centralized RL" (add "at equal budget" after EXP 1).

**D. Missing literature:** recent (2024–2026) RL/energy/fairness/hierarchical FL
selection — search and add verifiable refs.

**E. Missing implementation details:** PPO hyperparameters, network widths, sampling
scheme, environment/hardware, wall-clock overhead.

**F. Objections remaining after the *edit-only* changes:** the budget confound (needs
EXP 1) and the scaling weakness (needs EXP 3) — these cannot be edited away.

**G. Completable with NO new experiments (do this week):** items A1, A3–A10 except EXP
1/3; energy sensitivity, fairness metric, convergence/energy-to-target plots, PPO
table, related-work paragraph, all wording fixes.

**H. Absolutely requires new experiments:** EXP 1 (equal-budget), EXP 3 (scaling),
EXP 4 (reward sensitivity), EXP 6 (churn), EXP 5 cold-start. EXP 1 is the gatekeeper.

**I. Recommended structure:** Abstract → Introduction → Related Work → Problem
Formulation → HELM → Experimental Methodology (with the RQ→experiment→metric→evidence
table) → Results → Ablation & Sensitivity → Discussion → Limitations →
Reproducibility/Artifacts → Conclusion.

**J. Submission-readiness classification (evidence-based):**
- **As the manuscript stands today: PROMISING BUT MAJOR REVISION REQUIRED.** The
  seed wording and the budget confound are disqualifying for a careful reviewer.
- **After the edit-only fixes (Part 7-G): still MAJOR REVISION**, because EXP 1
  remains open.
- **After EXP 1 + sensitivity + convergence plots + related work, IF HELM still wins
  at equal budget: MEDIUM-CONFERENCE READY.** Add EXP 3/4 and it approaches
  STRONG-CONFERENCE READY for an edge-FL / systems-ML venue.
- I am **not** giving a numeric acceptance probability, per your instruction.

---

## PART 8 — CONCRETE PRIORITIZED ACTION PLAN

**Week 1 (no new training — pure credibility recovery):**
- [ ] Global find/replace the seed wording to the truth (critical).
- [ ] Soften the three overclaims (Pareto-optimal, "first," physical energy).
- [ ] Add PPO hyperparameter + environment table.
- [ ] From existing logs: energy-constant sensitivity table; participation-entropy +
      min-participation columns; accuracy-vs-rounds, accuracy-vs-cumulative-energy,
      energy-to-target plots.
- [ ] Draft the recent-related-work paragraph (after a real literature search).

**Week 2 (the one experiment that decides the paper):**
- [ ] EXP 1 equal-budget Centralized RL, all metrics + wall-clock + time-to-target.
      Report the result *whatever it is*; rewrite the causal claim to match.

**Week 3 (strengthen to medium/strong):**
- [ ] EXP 3 scaling 40→80→160; EXP 4 reward-weight grid; EXP 6 churn sweep;
      EXP 5 cold-start federation. Fold into Ablation & Sensitivity.

**Then:** re-run this reviewer pass. If EXP 1 holds and Week-1 fixes are in, the paper
moves from "major revision" to a defensible medium-venue submission.

---

### The single most important sentence in this review
Right now a reviewer can reject the paper with one line — *"the energy/fairness gains
over Centralized RL are confounded with selection budget (10 vs 14 clients/round) and
the '20 seeds' is not reflected in any table."* Fix the wording (an afternoon) and run
the equal-budget experiment (a day). Those two actions remove the two strongest,
fully-valid reasons to reject.
