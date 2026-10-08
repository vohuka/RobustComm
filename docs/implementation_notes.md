# Implementation Notes

How the paper was translated into code. Cross-references use the paper's section/theorem
numbering (`../main.pdf`).

## Design principles
- **NumPy-only core.** Aggregators, attacks, theory, and metrics use only numpy so tests
  run anywhere with no GPU/internet. Heavy MARL training on real environments is deferred
  to EC2 adapters.
- **Faithful, not padded.** Each module implements a specific paper object and cites it in
  its docstring. Where the paper is analysis-only (all numeric results are placeholders),
  the repo implements the *machinery* and demonstrates it on a synthetic environment rather
  than fabricating results.
- **Fail loud, never fake.** Operators raise when their Byzantine regime is violated;
  adapters raise until real data is provisioned; dummy baselines announce themselves.

## Robust aggregation (`aggregators/`, Def 3.1–3.2, Alg 1)
- `KrumAggregator` implements the Krum score `s(j)=Σ ||m_j−m_k||²` over the `n−f−2` nearest
  neighbours and selects the arg-min. Regime `2f+2<n` is enforced (`ValueError` otherwise).
- `MedianAggregator` / `TrimmedMeanAggregator` implement the coordinate-wise estimators;
  the median warns when `f/n≥1/2` (guarantee void) and the trimmed mean enforces a
  non-empty remainder.
- `MeanAggregator` is the DIAL/undefended baseline (breakdown point 0, Prop 9.1).

## Attack (`attacks/`, Def 5.1, Alg 2)
- `PerturbationSet` projects onto the `ε` `l∞`/`l2` ball (Eq. pertset).
- `FGSMMessageAttack` crafts perturbations on a **surrogate objective** (`SurrogateObjective`
  estimates the target `ŝ` via the robust centre of observed messages — a transfer-style,
  black-box estimate). The FGSM step is `δ ← Proj(δ + α·sign(∇ proxy_loss))`. `static` uses
  an analytic gradient through the mean; `adaptive` uses finite differences through the
  deployed `RobAgg`; `colluding` coordinates onto a shared direction.

## Theory (`theory/bounds.py`, Sec. 9)
Explicit functions for `β_K = 2√(n−f−2)+1` (Lem 9.2), the median `√d` factor (Lem 9.3), the
Theorem 9.4 per-step/episodic degradation bound (with the `γ=1 ⇒ factor=T` limit), the
Corollary 9.5 tolerance fraction/count, and the Proposition 9.6 tradeoff bound. These are
worst-case upper bounds (the paper states they are not tight); `test_theory_bounds.py`
checks the realised deviation never exceeds them.

## Synthetic environment (`envs/synthetic_comm.py`)
A minimal cooperative "recover the broadcast signal `s`" task: honest senders emit within an
`l2` ball of radius `B` (Assumption A1) of `s`; Byzantine senders emit adversarial messages;
agents aggregate their inbound neighbourhood (including their own honest message, so
broadcast neighbourhoods keep `n_i=n` and Krum stays valid) and are rewarded for recovering
`s`. This makes the paper's claim observable: the mean is pulled away by Byzantine senders
(reward drops) while robust aggregation holds. **It is not a paper benchmark.**

## Training (`training/adversarial.py`, Alg 3)
Alternates the inner FGSM attack with an outer linear-policy update (MSE toward the signal),
sampling one ensemble member per episode (MADDPG-style). A heuristic for the minimax
objective (Eq. minimax) with no convergence guarantee (Remark 9.8), exactly as stated.

## Baselines (`baselines/`, Sec. 10.5)
`BaselineSpec` encodes a defense configuration (operator + adversarial-training + ensembles);
`build_defense` constructs and optionally trains it. `standard_baselines` returns B1–B4 plus
the full RobustComm. Message signing (B5) and external defenses are adapter stubs with dummy
backends for tests.

## What was deliberately NOT done
- No real DIAL/MPE/SMAC training (EC2 adapters).
- No reproduction of the paper's placeholder tables/figures.
- No autograd/torch dependency (linear surrogate suffices for a faithful FGSM demo).
