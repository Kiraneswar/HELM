# FedRL-CS — Presentation Guide (3 Presenters)

**Project:** Federated Multi-Agent Reinforcement Learning for Hierarchical Client
Selection in Edge–Cloud Federated Learning Systems.

**How to use this document:** Persons 1, 2, and 3 each own one part below. Each part
has (a) a plain-language **script** you can say almost word-for-word, (b) **real-world
anchors** to make it concrete, (c) the **one figure to show**, and (d) **Q&A** for the
questions a professor is likely to ask. Read the "Shared 60-second pitch" first — all
three of you should be able to say it. Total talk time ≈ 15 min + Q&A.

---

## Shared 60-second pitch (everyone memorizes this)

> "Federated learning lets many devices — phones, hospitals, banks — train a shared AI
> model **without ever sending their private data to a server**. But you can't use every
> device every round: they have limited battery, weak networks, and drop in and out. So
> a central server must **choose which devices participate each round**. Our project puts
> a **reinforcement-learning agent in charge of that choice** — and, as our novel step,
> we let those selection agents, sitting on different edge servers, **share what they've
> learned with each other** on their own schedule. The honest result: it doesn't make the
> model *more accurate*, but it reaches the same accuracy while using **~6–7% less device
> energy** and treating devices **~15% more fairly** than a single centralized agent. That
> trade-off — same accuracy, cheaper and fairer — is the contribution."

**The one-line contribution:** *We federate the client-selection policy itself, not just
the model, and show it buys energy and fairness at no accuracy cost.*

---

## The map of the talk (so the audience knows where they are)

| Part | Presenter | Owns | Question it answers |
|------|-----------|------|---------------------|
| 1 | Person 1 | The problem & setup | *Why does client selection matter in the real world?* |
| 2 | Person 2 | Our method & novelty | *What did we actually build, and what's new?* |
| 3 | Person 3 | Results, rigor & future | *Did it work, how do we know, and where does it go?* |

---
---

# PART 1 — Person 1: The Problem (why this matters)

**Goal of your part:** make the audience *feel* the problem before anyone hears the word
"reinforcement learning." Spend ~4–5 minutes. End by handing over to Person 2 with:
"So the question becomes — *how* should we choose the devices? That's what [Person 2]
will explain."

### 1.1 What is Federated Learning (say this)

> "Normally, to train an AI model you collect everyone's data in one place. Federated
> learning flips that: the **model travels to the data**, not the other way around. A
> server sends the current model to your phone, your phone improves it a little using
> *your* data, and sends back **only the updated model weights — never your data**. The
> server averages everyone's updates into a new global model. Repeat. This is how Google's
> phone keyboard learns to predict your next word without Google ever reading what you
> type."

**Real-world anchors (name at least two):**
- **Google Gboard** — the phone keyboard that predicts your next word: the original
  large-scale real deployment of federated learning.
- **Apple** — QuickType / Siri improvements learned on-device.
- **Hospitals** — several hospitals jointly train a tumor-detection model without sharing
  patient scans (privacy law makes pooling the data illegal; federated learning is the
  workaround).
- **Banks / fintech** — fraud-detection and credit models trained across institutions that
  legally cannot share raw customer records. *(This is a strong, relatable example — data
  that is valuable but cannot leave the building.)*

> Founding reference you can name-drop: federated learning and the "FedAvg" averaging
> algorithm were introduced by Google (McMahan et al., 2017).

### 1.2 The catch: you can't use every device every round (say this)

> "Here's the practical problem. In the real world there might be *millions* of devices,
> but each round you can only afford to use a handful. Why? Because participating **costs
> the device**: it drains battery, it uses the user's mobile data, and it takes compute.
> On top of that, devices are unreliable — a phone goes into a tunnel, the battery dies,
> the app closes. So every round the server has to **pick a small subset of devices to
> participate**. That choice is called **client selection**, and it's the whole problem
> we work on."

**Make it concrete:**
- Picking **only the strongest devices** (newest phones, best wifi) → fast, but those
  users' data is not representative, and the same few devices get hammered every round —
  **unfair**, and it drains those specific phones.
- Picking **randomly** → fair, but you waste rounds on devices that drop out.
- Picking **everyone** → best data coverage, but maximum battery/energy cost and slowest.

> "So client selection is a **balancing act**: accuracy vs. energy vs. fairness. There's no
> obviously correct rule — which is exactly the kind of decision problem you can hand to a
> learning agent."

### 1.3 Our specific setting: edge–cloud with clusters (say this)

> "Real systems aren't flat. Devices connect through **edge servers** — think a cell tower,
> a regional data center, a hospital's local server. So we model a **two-tier system**:
> many devices grouped under a few **edge clusters**, and one **cloud** on top. In our
> experiments that's **40 devices split across 4 edge clusters**. Each cluster has its own
> local conditions — one cluster might be mostly weak mobile devices that drain fast,
> another mostly plugged-in strong devices."

**Figure to show:** *(draw or show a simple diagram)* Cloud on top → 4 edge clusters →
10 devices each. Mark one cluster as "weak/mobile," one as "strong/fixed."

### 1.4 Terms your teammates will use (define them now so Parts 2–3 are smooth)
- **Round** — one cycle of: pick devices → they train locally → server averages updates.
- **Non-IID data** — every device has *different* data (your photos ≠ my photos). Makes FL
  hard; we simulate it with a standard "Dirichlet" skew.
- **Straggler / churn** — devices that are slow or drop out mid-round.
- **Gini coefficient** — a 0-to-1 fairness score; **0 = everyone participates equally,
  1 = a few devices do all the work.** Lower is fairer. (Economists use it for income
  inequality — same idea, applied to which devices get used.)

### Part 1 — likely questions
- **"Why not just centralize the data?"** → Often illegal or impossible: privacy law
  (health, finance), data too large to move, or users won't consent. FL exists precisely
  for those cases.
- **"Isn't picking the strongest devices fine?"** → It biases the model toward those users
  and is unfair/expensive for them; weak devices (and their data) get permanently excluded.
- **"How is energy even measured?"** → *(Hand to Person 3 — it's a modeled cost in our
  simulation; be honest.)* "Person 3 covers exactly how we measure that."

---
---

# PART 2 — Person 2: Our Method & What's New

**Goal of your part:** explain the idea in layers, ending on the ONE novel thing. ~5 min.
Start by picking up Person 1's question: *"How should we choose the devices?"*

### 2.1 First layer — let an RL agent do the choosing (say this)

> "Instead of a fixed rule, we use **reinforcement learning**. Think of it as a coach that
> learns by trial and error. Each round the agent looks at the state of every device —
> battery, network speed, how recently it was used, how useful its data looked — and
> **picks which devices to include**. Then it gets a **reward**: high if the model improved
> a lot while using little energy and staying fair; low otherwise. Over many rounds it
> learns a selection strategy better than any hand-written rule. We use a standard, proven
> RL algorithm called **PPO**."

**Analogy:** "It's like a team manager choosing who plays each match — learning over a
season which line-ups win without exhausting the same star players every game."

### 2.2 Second layer — the standard part vs. the new part (say this)

> "Now, regular federated learning already shares something across devices: it shares the
> **model** (that's the averaging step). What nobody usually shares is the **selection
> strategy** — the agent's brain. In a two-tier system you'd normally have **one selection
> agent per edge cluster**, each learning on its own, in isolation."

> "**Our novel contribution:** we let those per-cluster agents **federate their selection
> policies too** — they periodically average their 'brains' with each other through the
> cloud, so a lesson one cluster learns (say, 'don't pick devices below 20% battery')
> spreads to all clusters, *without the clusters sharing any device data.*"

### 2.3 The two knobs that make it a real contribution (say this — this is the core)

> "Two design choices make this interesting and tunable:
>
> 1. **Two independent clocks.** The *model* is averaged every round (call it T_FL). The
>    *selection policy* is federated on its **own separate schedule**, every few rounds
>    (call it T_RL). Decoupling these two clocks is the technical heart of the paper —
>    the model and the decision-making strategy don't have to sync at the same rate.
>
> 2. **A blend dial, alpha (α).** When agents share brains, α controls the mix: α near 0
>    means 'fully adopt the shared consensus,' α = 1 means 'ignore the others, stay fully
>    local.' It lets us go from total collaboration to total independence with one number."

**Analogy for the two clocks:** "The *players* (model) get coached every single match, but
the *coaches* (selection agents) only meet to compare notes once a month. Different
cadences on purpose."

### 2.4 Who we compare against (say this — sets up Person 3)

> "To know if our idea helps, we compare it against four honest baselines:
> - **Random** selection — the simple sanity check.
> - **Full participation** — use every available device (the expensive gold standard for
>   data coverage).
> - **Oort** — a well-known *published* heuristic for smart client selection (Lai et al.,
>   2021), so we're not just beating strawmen.
> - **Centralized RL** — the most important one: a **single** RL agent that sees all
>   clusters at once. This is the real test: *does splitting the agent across clusters and
>   federating them beat just having one big central agent?* If we only tie with this, our
>   contribution is efficiency/decentralization, not raw performance — and that's exactly
>   what we found. Person 3 has the numbers."

**Figure to show:** the architecture diagram again, now with arrows showing (i) model
averaging every round and (ii) policy federation every T_RL rounds with blend α.

### Part 2 — likely questions
- **"Why PPO specifically?"** → It's stable and standard for this kind of discrete-choice
  policy; not the contribution, just a reliable tool.
- **"Doesn't sharing the policy leak data?"** → No — the policy is a decision strategy
  (weights of a small network), not the device data. Same privacy posture as sharing the
  model.
- **"Why decouple the two clocks — why not federate the policy every round too?"** → Policy
  learning is noisier and slower than model averaging; federating it every round can drown
  local adaptation. Separating the clocks lets each move at its natural rate — and our
  ablation tests exactly this (Person 3).
- **"What is the 'state' the agent sees?"** → Per-device features: battery, compute speed,
  bandwidth, how stale/recently-used it is, and a signal of how useful its data looked.

---
---

# PART 3 — Person 3: Results, Rigor & Future Scope

**Goal of your part:** deliver the honest result, prove you can be trusted, and paint the
future. ~5 min. The professor will respect **honesty + rigor** more than a big accuracy
number — lean into that.

### 3.1 The honest headline (say this first — don't bury it)

> "I'll be upfront: our method does **not** make the model more accurate. On accuracy it
> **ties** with the centralized agent and even with random selection — all within noise.
> We are **not** claiming an accuracy win, and that's deliberate. What we *do* show,
> consistently and reproducibly, is that we reach that same accuracy while using **less
> energy** and being **fairer**."

**The numbers (main comparison — 40 devices, 4 clusters, 100 rounds, averaged over 5 runs):**

| Method | Accuracy (%) | Energy (lower=better) | Fairness Gini (lower=fairer) |
|--------|:---:|:---:|:---:|
| **Federated MARL (ours)** | **63.31** | **19.61** | **0.49** |
| Random | 64.00 | 19.80 | 0.49 |
| Centralized RL | 63.30 | 20.99 | 0.58 |
| Full Participation | 63.16 | 22.54 | 0.58 |
| Oort (heuristic) | 61.45 | 19.74 | 0.58 |

> "Versus the centralized RL agent — our key comparison — same accuracy (63.31 vs 63.30),
> but **~6.6% less energy** and a Gini of 0.49 vs 0.58, which is **~15% fairer**."

**Figure to show:** `accuracy_energy_tradeoff.png` — our dot sits at the **far left**
(lowest energy) at the same height (accuracy) as the others. One picture = the whole story.

### 3.2 Why you should believe the numbers — the rigor story (say this — professors love this)

> "We were suspicious of our own good news, so we stress-tested it three ways:
>
> 1. **We changed a key training setting** (how often the agent updates) and re-ran
>    everything. The energy and fairness advantage held at **both** settings — ~6.6% and
>    ~6.3% energy, ~15% fairness both times. So it's not a fluke of one configuration.
>    *(Show `accuracy_tie_ue1_vs_ue3.png` — four accuracy lines lying on top of each other:
>    the tie is real.)*
>
> 2. **We independently re-checked the raw data**, not our own summary — confirming the
>    agent actually fired the exact number of times it should have, on every single run.
>
> 3. **The cross-check caught our own mistake** — two numbers we'd rounded the wrong way —
>    and we fixed them. We'd rather show you the number that survived scrutiny than the
>    prettiest one."

> "Also worth saying: this project went through several real bugs during development — a
> broken train/test split, a logging bug that faked results — and each was caught and
> fixed. The final numbers are the ones that held up *after* all of that."

### 3.3 The ablation — does the 'brain sharing' actually work? (say this)

> "We also asked: is the policy-federation mechanism even doing anything? We swept the
> blend dial α. When agents fully share (α small), their selection 'brains' become
> essentially **identical** — a similarity of ~1.0. When we turn sharing **off** (α = 1),
> similarity collapses to **~0.33** — they specialize to their local cluster. So the
> mechanism provably works and is controllable. Honestly, though: **α didn't move the final
> accuracy/energy/fairness** in our test setup — so we report it as *mechanism validation*,
> not as another performance lever."

**Figure to show:** `alpha_sweep.png` — left panel flat (task metrics), right panel the
similarity dropping off a cliff at α = 1. "The mechanism responds even where the outcome
doesn't — that's an honest, useful negative."

### 3.4 What this means in the real world (say this)

> "Translate the result out of the lab:
> - **~6–7% less energy** at the same accuracy means, across a million phones, meaningfully
>   less battery drained and less mobile data burned — directly a better user experience and
>   lower cost.
> - **~15% fairer participation** means weak and rural devices aren't permanently excluded,
>   so the model actually represents *everyone*, not just people with new phones and good
>   wifi. That's an equity and model-quality win at once.
> - And crucially, the smarter selection is learned **per-region but shared globally**,
>   without any region exposing its users' data."

### 3.5 Honest limitations (say this — don't hide it, it builds credibility)

> "Two honest caveats. First, our data and our energy/battery/network conditions are
> **simulated** — a controlled synthetic benchmark, not a real device fleet yet — so the
> exact percentages are indicative, not field-measured. Second, these are small-scale runs
> (tens of devices). Both are the natural next steps."

### 3.6 FUTURE SCOPE — where this research goes (this is your strong finish)

Frame these as "what we'd do next, and why each matters in the real world."

1. **Real datasets & real deployment.** Swap the synthetic task for standard federated
   benchmarks (e.g., FEMNIST/Shakespeare, or real tabular data) and eventually a real
   device fleet, so the energy and fairness gains are field-measured, not modeled.
   *Real-world payoff: evidence a company could act on.*
2. **A grounded energy model.** Replace our modeled battery-drain with measured device
   power profiles, so "6–7% energy" becomes real joules and rupees. *Payoff: a concrete
   cost/carbon number.*
3. **Scale up.** Thousands-to-millions of devices and more clusters — the regime real
   deployments (Gboard-scale) actually live in.
4. **Add privacy guarantees.** Layer in differential privacy / secure aggregation so even
   the shared *policy* is provably private. *Payoff: deployable under strict privacy law
   (health, finance).*
5. **Personalization.** Let each cluster keep a personalized policy *and* benefit from
   sharing — tune α adaptively instead of fixing it.
6. **Domain-specific impact.** Two concrete verticals:
   - **Healthcare** — hospitals collaboratively selecting which sites contribute each round
     under strict energy/compute budgets, without moving patient data.
   - **Finance / lending** — banks and fintech lenders training shared fraud or
     credit-risk models across institutions that legally cannot pool customer records;
     smart, fair client selection decides which institution's edge participates each round.
7. **Theory.** Convergence guarantees for the two-clock (decoupled T_FL / T_RL) scheme —
   turning the empirical finding into a proof.

### 3.7 Closing line (say this)

> "So to sum up: we don't claim a smarter model — we claim a **smarter, cheaper, and fairer
> way to build the same model**, by letting regional selection agents learn locally and
> share what they learn globally. And we've been rigorous and honest about exactly what
> does and doesn't hold. Thank you — happy to take questions."

### Part 3 — likely questions
- **"Your accuracy isn't higher — so why is this useful?"** → Because at deployment scale,
  energy and fairness *are* the bottleneck, not the last decimal of accuracy. Same model,
  less battery, fairer inclusion, is a real operational win.
- **"63% accuracy sounds low."** → It's a deliberately hard synthetic task tuned so methods
  are *separable*; absolute accuracy isn't the point, the *comparison between methods* is.
  (The ablation runs plateau ~49% for the same reason — smaller setup.)
- **"How do you know it's fair — what's Gini?"** → 0 = every device used equally, 1 = a few
  do all the work. We go from 0.58 (centralized) to 0.49 — meaningfully more balanced.
- **"Is the energy number real?"** → Honest answer: it's a modeled cost in our simulation
  (battery-drain per selected device); making it field-measured is future work #2.
- **"What stops one cluster's bad policy from poisoning the others?"** → The blend dial α
  and the slow federation clock damp any single cluster's influence; robustness to
  adversarial clusters is a good future direction.

---
---

## Appendix A — Timing & who shows which figure

| Part | Presenter | Minutes | Figure(s) to show |
|------|-----------|:---:|-------------------|
| Pitch | all / Person 1 opens | 1 | — |
| 1 | Person 1 | 4–5 | Architecture diagram (cloud → 4 clusters → devices) |
| 2 | Person 2 | 5 | Same diagram + arrows for model-avg vs policy-federation; the two clocks |
| 3 | Person 3 | 5 | `accuracy_energy_tradeoff.png`, `accuracy_tie_ue1_vs_ue3.png`, `alpha_sweep.png` |
| Q&A | all | 3–5 | keep `PAPER_NUMBERS.md` open |

## Appendix B — 10-word glossary (for quick reference during Q&A)
- **Federated learning** — train a shared model without moving anyone's data.
- **Client selection** — choosing which devices participate each round.
- **PPO** — the reinforcement-learning algorithm that makes the choice.
- **T_FL / T_RL** — the two separate clocks (model averaging vs policy sharing).
- **alpha (α)** — the collaborate-vs-stay-local blend dial.
- **Gini** — fairness score, 0 = fair, 1 = unfair (lower is better).
- **Non-IID** — every device has different data.
- **Centralized RL** — the one-big-agent baseline we tie with (and beat on cost/fairness).

## Appendix C — The three sentences that must land
1. We put an RL agent in charge of choosing devices, and — new — we let those agents share
   what they learn across regions without sharing data.
2. It doesn't raise accuracy; it reaches the same accuracy with ~6–7% less energy and ~15%
   better fairness than a centralized agent, and that holds under stress-testing.
3. The mechanism provably works (we can dial it), we caught our own bugs, and the honest
   next step is real data and a measured energy model.
