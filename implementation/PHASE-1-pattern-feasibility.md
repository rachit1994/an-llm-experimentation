# Phase 1 — Pattern feasibility (the G0 gate)

**Goal:** answer the one question that can kill the project — *does predicting the next **pattern**
train without collapsing, at competitive perplexity, against a fair baseline?* Everything else waits
on this. Estimate: this plus Phase 0 is the **~3–4 week** kill-or-prove-the-core milestone.

Depends on: Phase 0 (a gradient-checked `Complex` stack + byte data + fixed splits).
Blocks: Phases 2, 3, 4 (and thus everything).

---

## What we build

1. **A next-pattern predictor.** Encoder maps byte context → pattern `z`; predictor guesses
   `ẑ_{n+1}`; loss `L_pattern = D(ẑ_{n+1}, stopgrad(z_{n+1}))` (doc 07). Anti-collapse via one of:
   - **JEPA** (EMA target encoder) **+ VICReg** (variance + covariance terms), or
   - **InfoNCE** (contrastive with negatives), or
   - **VQ codebook** (predict a code index; EMA codebook + dead-code reinit).
   All three are switches (C6); we run all three.
2. **A param-matched next-token baseline (C7).** Same body size, same data (raw bytes, C2), same
   loop, same tuning budget, param-matched to ±5% by the automated counter. Trained with ordinary
   cross-entropy.
3. **Collapse instrumentation — built FIRST, before any training curve is trusted:**
   - **Effective rank** of the predicted-pattern matrix over the val set (participation ratio of
     singular values, `(Σσ_i)² / Σσ_i²`).
   - **Per-dimension variance** of predicted patterns (vs a constant-predictor baseline).
   - **Codebook usage/perplexity** if the VQ variant is on.

---

## Metrics

- **Collapse (H3):** effective rank and per-dim variance, logged every epoch.
- **Perplexity:** the pattern model is decoded to symbols through a **fixed decoder** so it is
  comparable to the token baseline's perplexity on the same val set (apples-to-apples).
- **Cost:** wall-clock, peak RAM, params — for the record and the Pareto curve (doc 06).

---

## The G0 gate (project go/no-go)

```
PASS (G0)  ⟺   effective rank ≥ 50% of embedding dim
               AND per-dim variance ≥ 10× a constant predictor      (no collapse, H3)
               AND val perplexity ≤ 1.10 × param-matched token baseline   (competitive, H2)

FAIL       ⟹   STOP THE PROJECT.
```

The kill numbers are hard: **effective rank < 50% of embed dim**, or **perplexity > 110%** of the
baseline, ends the project. There is no "directionally encouraging" pass — collapse or a >10%
perplexity gap is a stop, because the pattern objective is the load-bearing leg (doc 05).

---

## Order of operations (this order matters)

1. **Wire the collapse metrics first**, and prove them on a *deliberately collapsed* toy model
   (constant encoder) — the metrics must *scream* on the toy before you trust them on the real run.
2. Train the **token baseline** to convergence; record its perplexity and cost. *This is the number
   to beat, and it must exist before the pattern model is judged.*
3. Train the **pattern model** (each anti-collapse variant), watching the collapse metrics *live*.
4. Compare at the gate. Report mean ± σ over ≥5 seeds (doc 06); a "pass" inside 1σ of the baseline
   is a pass, not a win — G0 only asks *competitive*, not *better*.

---

## Anti-patterns (how Phase 1 lies to you)

- **A loss that plunges to ~0 in a few hundred steps** → collapse, not success (doc 07 scar). The
  collapse metrics, not the loss curve, are the truth.
- **Judging the pattern model without a converged baseline** → you have no denominator; the
  perplexity gate is meaningless. Baseline first.
- **Comparing at unequal params** → C7 violation; the automated counter gates the run.
- **Reporting one lucky seed** → seed noise can hide collapse or fake competitiveness; ≥5 seeds.

---

### Interview questions this doc answers

- *"What's the single experiment that decides the project?"* Phase 1's G0 gate: does the
  next-pattern predictor avoid collapse (effective rank ≥ 50%, variance ≥ 10× constant) at ≤ 110%
  of the param-matched token baseline's perplexity?
- *"How do you know it didn't just collapse?"* Effective-rank + per-dim-variance instrumentation
  wired *before* training and proven on a deliberately-collapsed toy first; the loss curve is not
  trusted.
- *"Why is the baseline built first?"* Without a converged, param-matched token baseline there is no
  denominator for the perplexity gate (C7).

### Operator's scar

We once ran the pattern model for a week, got a gorgeous loss curve, and only wired the
effective-rank metric at the end — it read **1.0** (fully collapsed) the whole time. A week, gone, on
a model that had learned a constant. The scar is the non-negotiable order: *collapse metrics first,
proven on a collapsed toy, before a single real training step is believed.*
