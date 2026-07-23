# Phase 1 — Pattern feasibility (the G0 gate)

**Goal:** answer the one question that can kill the project — *does predicting the next **pattern**
train without collapsing, at competitive perplexity, against a fair baseline?* Everything else waits
on this. Estimate: this plus Phase 0 is the **~3–4 week** kill-or-prove-the-core milestone.

Depends on: Phase 0 (a gradient-checked `Complex` stack + byte data + fixed splits).
Blocks: Phases 2, 3, 4 (and thus everything).

---

## What we build

1. **A next-pattern predictor with an explicit invariance term.** Encoder maps byte context →
   pattern `z`; predictor guesses `ẑ_{n+1}`; loss
   `L = L_pattern + L_inv + anti-collapse + λ·L_byte-CE`, where `L_pattern = D(ẑ_{n+1}, stopgrad(z_{n+1}))`
   (doc 07) and `L_inv = D(enc(aug(x)), stopgrad(enc(x)))` **trains** the headline invariance (protocol
   §4.4) — without it, next-pattern prediction alone would not produce augmentation-invariance, and
   the plan would test a property it never trained for. `L_inv` is a switch (`no_invariance`) whose
   ablation must fail the H4 gate (proving the invariance is caused by the objective). Anti-collapse
   via one of:
   - **JEPA** (EMA target encoder) **+ VICReg** (variance + covariance terms), or
   - **InfoNCE** (contrastive with negatives), or
   - **VQ codebook** (predict a code index; EMA codebook + dead-code reinit).
   All three are switches (C6); we run all three.
2. **A param-matched next-token baseline (C7).** Same body size, same data (raw bytes, C2), same
   loop, same tuning budget, param-matched to ±5% by the automated counter. Trained with ordinary
   cross-entropy.
3. **Collapse instrumentation — built FIRST, before any training curve is trusted:**
   - **Effective rank** of the predicted-pattern matrix `Z` over the val set (participation ratio of
     singular values, `(Σσ_i)² / Σσ_i²`) — compared to the **target** patterns' effective rank.
   - **Mean per-dimension std** of `Z` — compared to the **target** patterns' per-dim std.
   - **Codebook usage/perplexity** if the VQ variant is on.

   Exact thresholds and the *reason the comparison is against the target* (not a constant) are pinned
   in [`METRICS-AND-GATES.md`](METRICS-AND-GATES.md) §2.

---

## Metrics

- **Collapse (H3):** effective rank and mean per-dim std of predictions **relative to the targets**
  (protocol §2), logged every epoch.
- **Quality — bits-per-byte (BPB), not "perplexity via a decoder".** A next-pattern predictor has no
  native token distribution, so "perplexity" was under-specified. Both arms instead emit a next-**byte**
  distribution and are scored by **BPB** on the same held-out bytes: the token baseline via its softmax,
  the pattern model via a **256-way Born readout head** (`256·d ≈ 0.13`M params, negligible). Exact
  definition in protocol §1.
- **Cost:** wall-clock, peak RAM, params — for the record and the Pareto curve (doc 06).

---

## The G0 gate (project go/no-go)

Exact definitions and the pre-registered constants live in
[`METRICS-AND-GATES.md`](METRICS-AND-GATES.md) §2–§3; the gate is:

```
PASS (G0)  ⟺   erank(Z)   ≥ 0.50 · erank(Z_target)          (no collapse, H3 — §2)
               AND meanstd(Z) ≥ 0.50 · meanstd(Z_target)    (no collapse, H3 — §2)
               AND BPB_deploy(pattern) ≤ 1.10 · BPB_deploy(token baseline)   (competitive, H2 — §1,§3)
               over 5 seeds (mean).

FAIL       ⟹   STOP THE PROJECT.
```

Two corrections over the first draft, both in protocol §2–§3: (1) collapse is measured **relative to
the target patterns** — the earlier "≥ 10× a constant predictor" was vacuous, since a constant has
variance `0` and `10·0 = 0` passes everything; (2) "perplexity" is now **BPB**, which is exactly
computable and tokenizer-free (C2). The kill numbers are hard and pre-registered: predictions spanning
< half the target's effective rank or per-dim spread, **or** BPB > 110% of the baseline, ends the
project. There is no "directionally encouraging" pass.

**The improvement, stated without if-and-but (protocol §3).** G0 is *feasibility*, not the improvement.
The improvement is two exact claims: (a) **iso-`(d,L)`**: the pattern model has **2.28× fewer params**
by construction (param counter, no training needed); (b) **iso-parameter**: given the reclaimed budget
back, `BPB(pattern) < BPB(token) − 0.02` bits/byte, paired over 5 seeds, `p<0.05` — the one
unconditional "it is better" claim. If (b) fails we claim only (a), which is arithmetic.

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
  next-pattern predictor avoid collapse (effective rank and per-dim std each ≥ 0.50× the *target's*)
  at BPB ≤ 1.10× the param-matched token baseline (protocol §2–§3)?
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
