# FedRL-CS — Limitations & How to Improve

This is the honest, complete limitation list for the project **as it exists today**,
with a concrete fix for each. The first four are the ones the team already identified;
items 5–10 are additional real limitations found by reading the code — some sharp enough
that a careful reviewer/professor could raise them, so it's better to own them first.
A prioritized roadmap is at the end.

Legend: **Effort** = rough work to fix (Low/Med/High). **Impact** = how much it
strengthens the paper.

---

## The four already identified

### L1 — Synthetic data  *(Effort: Med · Impact: High)*
We use scikit-learn `make_classification`, not a real dataset or device fleet.
- **Why it matters:** reviewers can't tell if the gains transfer to real, messy data.
- **Fix:** swap the data layer (`src/data/generator.py`) for a standard FL benchmark —
  FEMNIST or Shakespeare via the LEAF suite, or a real tabular dataset (e.g., a public
  credit-risk / fraud set, which also fits an edge-lending story). The FL loop, RL, and
  metrics all stay the same; it's mostly a loader swap plus a CNN/embedding model for
  image/text.
- **Highest value-per-hour change in the whole list.**

### L2 — Simulated energy  *(Effort: Med · Impact: High)*
Energy is a modeled accumulator: `energy += drain_per_round` for each selected client,
with `drain_per_round` drawn from a hand-picked Gamma distribution.
- **Why it matters:** "6–7% less energy" is currently a unitless proxy, not joules.
- **Fix:** replace the drain constant with a cited device-power model —
  energy ≈ (compute cost per local epoch) + (bytes uploaded × radio energy-per-bit),
  using published smartphone/edge power profiles. Then energy is reported in real units
  and the % becomes defensible.

### L3 — Small scale (40 devices)  *(Effort: Med · Impact: Med-High)*
Real deployments have thousands–millions of clients; we use 40 across 4 clusters.
- **Why it matters:** selection matters *more* at scale (you sample a tiny fraction), so
  small scale may understate the benefit.
- **Fix:** scale to hundreds/thousands of clients with per-round subsampling; add more
  clusters. Report communication rounds and wall-clock vs. scale so the scaling trend is
  explicit. Watch RL stability as the action space grows (may need action factorization).

### L4 — Simplified RL setup (independent PPO)  *(Effort: High · Impact: Med-High)*
Each cluster runs its own PPO agent with no shared credit assignment.
- **Why it matters:** independent learners can work at cross-purposes and learn slowly;
  it's not "true" multi-agent RL.
- **Fix:** move to a credit-assignment MARL method — MAPPO with a centralized critic,
  QMIX (value factorization), or COMA (counterfactual baselines). Keep the federated
  policy-sharing idea on top; the novelty then sits on a stronger RL backbone.

---

## Additional limitations found in the code (own these too)

### L5 — The network/straggler simulation does not affect results  *(Effort: Low · Impact: High)* ⚠️ sharpest
`runner.py` runs a SimPy upload/deadline/dropout model and logs `num_arrived`, but then:
```python
# runner.py ~line 427
arrived_updates = client_updates   # keep all for aggregation
```
i.e. **every trained update is aggregated regardless of whether it arrived on time.**
Latency, bandwidth contention, and deadline misses currently change nothing downstream.
- **Why it matters:** a reviewer who reads the loop will notice the network model is
  decorative; it undercuts the "realistic edge conditions" framing.
- **Fix:** actually filter — aggregate only `arrived_ids` that beat the round deadline;
  optionally give the RL agent a penalty for selecting clients that then time out. Small
  code change, large credibility gain. (This is the single cheapest high-impact fix.)

### L6 — The energy/fairness "win" is partly built into the reward  *(Effort: Low · Impact: High)*
The reward explicitly rewards low energy and low Gini (weights `accuracy 1.0 / energy 0.1
/ fairness 0.05`). So "our method saves energy and is fairer" is partly *by construction*.
- **Why it matters:** obvious professor question — *"Isn't it circular? You rewarded it
  for exactly the metrics you now report as wins."*
- **Fix:** run the **reward-weight ablation** (the config already supports it): set energy
  and fairness weights to 0 and show what happens; sweep them to show the trade-off
  frontier. The honest, strong version of the claim is *"with a modest cost weight, the
  federated policy reaches the same accuracy at lower energy/Gini than a centralized agent
  under the identical reward,"* — the comparison is apples-to-apples because every RL
  method sees the same reward.

### L7 — Evidence the RL is actually learning is weak  *(Effort: Med · Impact: Med-High)*
Accuracy ties with **Random**, and the α and T_RL ablations were statistically flat —
both consistent with the policy not discovering much beyond a mild heuristic on this task.
- **Why it matters:** if Random ties you on accuracy *and* energy is reward-driven, a
  reviewer may ask what the RL contributes at all.
- **Fix:** (a) add a **learning curve** — plot reward / selection quality over rounds to
  show the policy improves; (b) use a **harder, selection-sensitive benchmark** (stronger
  non-IID, a tighter energy budget so bad selections are punished) where smart selection
  clearly beats random. If, on a sensitive benchmark, RL still only ties random, report
  that honestly — it's a real finding about when selection matters.

### L8 — Statistically underpowered  *(Effort: Low · Impact: Med)*
5 seeds (main), 3 seeds (ablation), with accuracy std ≈ 2–4 pp. That's *why* accuracy
differences are indistinguishable — but it also weakens the energy/fairness claims.
- **Fix:** run **15–20 seeds** and report a significance test (paired t-test or bootstrap
  confidence intervals) on the energy and Gini gaps, so "−6.6% energy" carries a CI/p-value.
  Cheap compute, directly strengthens the headline claim.

### L9 — Privacy is claimed but not implemented  *(Effort: Med · Impact: Med)*
The framing says sharing the *policy* (not data) preserves privacy, but there is no
differential privacy or secure aggregation in the code, and sharing policy weights can in
principle leak information.
- **Fix:** add DP noise or secure aggregation to the policy-federation step and measure the
  privacy/utility trade-off; **or** soften the claim to *"same privacy posture as standard
  FedAvg; formal guarantees are future work."* Don't over-claim privacy without a mechanism.

### L10 — Federation collapses agents to (near-)identical policies  *(Effort: Med · Impact: Med)*
Policy cosine similarity → ~1.0 for α ≤ 0.7 (only α = 1.0 keeps them distinct at ~0.33).
- **Why it matters:** if all clusters' agents become identical, what does the *multi-agent*
  structure add over one shared agent — which is exactly the centralized-RL baseline you
  tie with? It's a genuine conceptual tension.
- **Fix:** either (a) reframe the contribution as **decentralized, privacy-preserving
  training** (no raw data or gradients centralized) rather than per-cluster specialization,
  or (b) add **personalization** — keep a local policy head per cluster and only federate a
  shared backbone — so agents stay usefully different while still sharing. Then show a
  regime where the personalized-federated version beats both fully-shared and fully-local.

---

## Smaller / methodological notes (mention only if asked)
- **Fixed hand-tuned hyperparameters** (lr, PPO epochs, clip, α, T_RL) — a small sweep or
  sensitivity table would help.
- **Static topology** — clients don't move between clusters and clusters don't appear/vanish
  mid-run; real edge systems are more dynamic.
- **Communication cost of federating the policy itself is not accounted** — sharing policy
  weights has its own (small) cost that isn't in the energy ledger.
- **Single model family (small MLP)** — no CNN/transformer; conclusions are for this model
  class until L1 broadens it.

---

## Prioritized roadmap (biggest credibility per hour)

**Do first (Low effort, High impact) — before the work is judged seriously:**
1. **L5** — make the network sim actually filter aggregation (fixes a real gap in the loop).
2. **L6** — run the reward-weight ablation to defuse the "circular reward" critique.
3. **L8** — more seeds + confidence intervals on the energy/Gini gaps.

**Do next (Med effort, High impact) — the substance upgrades:**
4. **L1** — a real FL benchmark dataset.
5. **L2** — a grounded, cited energy model (report joules).
6. **L7** — learning curves + a selection-sensitive benchmark to prove the RL earns its keep.

**Bigger research steps (High effort) — the "future work" section:**
7. **L4** — a credit-assignment MARL backbone (MAPPO/QMIX/COMA).
8. **L3** — scale to thousands of clients.
9. **L9 / L10** — real privacy guarantees, and personalization so federation adds value
   beyond a single shared agent.

**One-line honest positioning for tomorrow:** *"This is a controlled proof-of-concept: the
mechanism works and the energy/fairness trade-off is consistent, but the data, energy model,
and scale are simplified, and the network model and reward design each have a known gap we've
scoped concrete fixes for."* Owning the list is more convincing than hoping no one asks.
