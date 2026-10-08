# PAPER_NUMBERS.md — numbers for the Results section (Section VI)

Every value below is pulled from `summary/comparison_table.csv` (robustness) and
`summary/ablation_results.csv` (ablations), rounded to at most 2 decimal places.
Values are quoted as `mean ± std` across seeds. The only figures NOT taken verbatim
from those two files are in the clearly marked **Derived** blocks, which are simple
arithmetic on the rows directly above them (formula shown each time).

> Note: the paper source `fedrl_cs_paper.tex` was not in the connected folder, so the
> table column order here follows a standard IEEE Results layout. Confirm the headers
> match your actual Section VI table before pasting; the numbers themselves are final.

Fixed experimental setup (context, not results): main comparison = 40 clients /
4 clusters / 100 rounds / 5 seeds; ablations = 20 clients / 3 clusters / 50 rounds /
3 seeds. `Energy` = cumulative sum of per-round energy over the run. `Gini` = final-round
participation Gini (lower = fairer). `update_every` = PPO gradient-flush cadence.

---

## Table I — Main comparison (update_every = 1, 5 seeds)
Source: `comparison_table.csv`, rows with `update_every = 1`.

| Method | Accuracy (%) | Energy | Gini |
|--------|--------------|--------|------|
| Federated MARL (Ours) | 63.31 ± 2.19 | 19.61 ± 0.83 | 0.49 ± 0.02 |
| Random | 64.00 ± 1.57 | 19.80 ± 0.87 | 0.49 ± 0.02 |
| Centralized RL | 63.30 ± 2.63 | 20.99 ± 0.80 | 0.58 ± 0.01 |
| Full Participation | 63.16 ± 2.64 | 22.54 ± 1.11 | 0.58 ± 0.01 |
| Oort (Heuristic) | 61.45 ± 3.01 | 19.74 ± 0.99 | 0.58 ± 0.03 |

## Table II — Robustness: FMARL vs Centralized RL at both PPO cadences
Source: `comparison_table.csv`, all rows for these two methods.

| Method | update_every | Accuracy (%) | Energy | Gini |
|--------|:---:|--------------|--------|------|
| Federated MARL (Ours) | 1 | 63.31 ± 2.19 | 19.61 ± 0.83 | 0.49 ± 0.02 |
| Federated MARL (Ours) | 3 | 63.32 ± 1.44 | 19.68 ± 0.92 | 0.49 ± 0.02 |
| Centralized RL | 1 | 63.30 ± 2.63 | 20.99 ± 0.80 | 0.58 ± 0.01 |
| Centralized RL | 3 | 63.42 ± 2.58 | 21.00 ± 0.77 | 0.58 ± 0.02 |

**Derived** (arithmetic on the Table II rows; formulas shown, not raw CSV cells):
- Energy reduction, FMARL vs CRL = (CRL_energy − FMARL_energy) / CRL_energy:
  - update_every = 1: (20.99 − 19.61) / 20.99 = **6.57 %**
  - update_every = 3: (21.00 − 19.68) / 21.00 = **6.28 %**
- Gini reduction, FMARL vs CRL = (CRL_gini − FMARL_gini) / CRL_gini:
  - update_every = 1: (0.58 − 0.49) / 0.58 = **15.44 %**
  - update_every = 3: **15.37 %**
- Accuracy gap, FMARL − CRL:
  - update_every = 1: 63.31 − 63.30 = **+0.01 pp**
  - update_every = 3: 63.32 − 63.42 = **−0.10 pp**  (sign flips; both far inside ±2 pp std)

## Table III — Alpha ablation (federated_marl, update_every = 1, T_RL = 5, 3 seeds)
Source: `ablation_results.csv`, `sweep = alpha`.

| alpha | Accuracy (%) | Energy | Gini | Policy sim (last fed.) |
|:---:|--------------|--------|------|:---:|
| 0.0 | 48.96 ± 2.32 | 9.55 ± 0.55 | 0.38 ± 0.02 | 1.00 |
| 0.3 | 49.29 ± 2.80 | 9.56 ± 0.53 | 0.38 ± 0.02 | 1.00 |
| 0.5 | 49.25 ± 2.83 | 9.56 ± 0.53 | 0.38 ± 0.02 | 1.00 |
| 0.7 | 49.25 ± 2.71 | 9.52 ± 0.59 | 0.38 ± 0.02 | 1.00 |
| 1.0 | 50.08 ± 4.17 | 9.54 ± 0.59 | 0.38 ± 0.02 | 0.33 ± 0.00 |

(`policy sim` at alpha 0.7 is 0.999 in the CSV → 1.00 at 2 dp; the mechanism signal is
the drop to 0.33 at alpha = 1.0, the no-federation setting.)

## Table IV — T_RL cadence ablation (federated_marl, update_every = 1, alpha = 0.5, 3 seeds)
Source: `ablation_results.csv`, `sweep = T_RL`.

| T_RL | Accuracy (%) | Energy | Gini | Policy sim (last fed.) |
|:---:|--------------|--------|------|:---:|
| 1 | 49.75 ± 1.35 | 9.44 ± 0.53 | 0.38 ± 0.01 | 1.00 |
| 3 | 49.50 ± 2.20 | 9.55 ± 0.52 | 0.38 ± 0.02 | 1.00 |
| 5 | 49.25 ± 2.83 | 9.56 ± 0.53 | 0.38 ± 0.02 | 1.00 |
| 10 | 49.42 ± 2.80 | 9.55 ± 0.52 | 0.38 ± 0.02 | 1.00 |

(`policy sim` at T_RL 10 is 0.9988 in the CSV → 1.00 at 2 dp.)

---

## Claim guardrails (locked findings)
- **Primary empirical claim:** vs Centralized RL, Federated MARL cuts energy by ~6.6 %
  (update_every 1) / ~6.3 % (update_every 3) and lowers participation Gini by ~15 %,
  consistent at both cadences (Table II + Derived).
- **Do NOT claim any accuracy advantage.** The FMARL−CRL accuracy gap is +0.01 pp at
  update_every 1 and −0.10 pp at update_every 3 — the sign flips and both are within
  noise (Table II).
- **Alpha ablation = mechanism validation, not an outcome improvement.** Policy
  similarity collapses to 0.33 at alpha = 1.0 vs ~1.00 elsewhere (Table III), confirming
  federation works; but accuracy/energy/Gini do not move materially across alpha or T_RL
  (Tables III–IV).

## LaTeX (Table I and II, numbers identical to above)
```latex
% Table I — main comparison, update_every = 1
\begin{tabular}{lccc}
\hline
Method & Accuracy (\%) & Energy & Gini \\
\hline
Federated MARL (Ours) & 63.31 $\pm$ 2.19 & 19.61 $\pm$ 0.83 & 0.49 $\pm$ 0.02 \\
Random                & 64.00 $\pm$ 1.57 & 19.80 $\pm$ 0.87 & 0.49 $\pm$ 0.02 \\
Centralized RL        & 63.30 $\pm$ 2.63 & 20.99 $\pm$ 0.80 & 0.58 $\pm$ 0.01 \\
Full Participation    & 63.16 $\pm$ 2.64 & 22.54 $\pm$ 1.11 & 0.58 $\pm$ 0.01 \\
Oort (Heuristic)      & 61.45 $\pm$ 3.01 & 19.74 $\pm$ 0.99 & 0.58 $\pm$ 0.03 \\
\hline
\end{tabular}

% Table II — robustness, FMARL vs CRL at update_every = 1 and 3
\begin{tabular}{llccc}
\hline
Method & $u$ & Accuracy (\%) & Energy & Gini \\
\hline
Federated MARL (Ours) & 1 & 63.31 $\pm$ 2.19 & 19.61 $\pm$ 0.83 & 0.49 $\pm$ 0.02 \\
Federated MARL (Ours) & 3 & 63.32 $\pm$ 1.44 & 19.68 $\pm$ 0.92 & 0.49 $\pm$ 0.02 \\
Centralized RL        & 1 & 63.30 $\pm$ 2.63 & 20.99 $\pm$ 0.80 & 0.58 $\pm$ 0.01 \\
Centralized RL        & 3 & 63.42 $\pm$ 2.58 & 21.00 $\pm$ 0.77 & 0.58 $\pm$ 0.02 \\
\hline
\end{tabular}
```
