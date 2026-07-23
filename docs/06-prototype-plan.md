# 06 — Prototype plan (a fair, same-machine bake-off)

The prototype's job is to produce **attributable** numbers on an M1, under the charter. This doc
specifies the cuts, the fairness rules, and — crucially — what *"test against LLMs"* actually means
(it is **not** a head-to-head against GPT).

---

## The three cuts

### Cut P — Pattern-as-token (the lead, collapse-instrumented)

The most valuable and most killable cut, run **first** (it is Phase 1). A next-**pattern** predictor
vs a param-matched next-**token** baseline, with **collapse instrumentation on from line one**
(effective rank + per-dim std of predictions, relative to the targets). Data: raw bytes/chars (C2).
Gate: **G0** (no collapse; BPB ≤ 1.10× the byte baseline — exact protocol in
[`../implementation/METRICS-AND-GATES.md`](../implementation/METRICS-AND-GATES.md)). See
[`07-pattern-as-token-the-objective.md`](07-pattern-as-token-the-objective.md).

### Cut A — AG News classification (the "known-winnable" cut)

The task where the prior art is strongest (Wave Network 90.91/91.66% single-layer). We reproduce the
*comparison* under our fairness rules: our own complex layer vs our own param-matched real layer,
both from-scratch (no BERT embeddings, C1/C7). Purpose: establish that **complex phase earns its
keep** (H1) on a task we already expect it to, before betting on generation. Metric: test accuracy
+ the phase=0 ablation gap.

### Cut B — Generation (the honest hard case)

Char/byte-level next-symbol generation with **Born readout**, explicitly probing the `O(|V|·d)`
wall and its mitigations (low-rank/tied/hierarchical/VQ). Purpose: measure *how far* generation is
from parity, not to declare victory. This is where we expect to **narrow to encoder-only** if the
wall holds (doc 05, Phase 6).

---

## Fairness rules (these are what make the numbers mean anything)

1. **Equal params (±5%).** The complex model and its baseline have matched parameter counts,
   verified by an automated counter, *before* any run (the C7 scar, GROUND-UP-CONSTRAINTS).
2. **Equal tuning budget.** Same number of hyperparameter-search trials, same search space size,
   same early-stopping rule, for both arms. A win from an unequal tuning budget is not a win.
3. **Equal data, splits, and loop.** Same train/val/test (fixed, hashed, leak-checked — C5), same
   optimizer, same schedule, same compute budget, same seeds swept.
4. **One switch at a time (C6).** Every comparison toggles exactly one mechanism. Named ablations:
   - `phase = 0` (real-valued) — the decisive H1 ablation.
   - `no_unitary` (free linear map instead of unitary).
   - `no_interference` / `no_modulation`.
   - `no_glow`.
   - `bpe` (subword instead of bytes — an ablation, never a default, C2).
5. **Report with config (C4).** Every number ships with seed, versions, git SHA, and reproduces
   bit-for-bit on CPU.
6. **Seeds are a distribution, not a point.** Report mean ± σ over ≥5 seeds; a "win" smaller than
   1σ of the seed spread is noise, not a result (this is the H1 kill threshold).

---

## What "test against LLMs" means (read this before quoting a comparison)

It does **NOT** mean a head-to-head against GPT-4/Claude/Llama. That comparison is
**uninformative and unfair**: those models have orders of magnitude more parameters, pretraining
data, and compute; losing to them tells us nothing, and any framing that implies we might beat them
is the credibility-killer doc 05 warns about.

It **DOES** mean three legitimate, apples-to-apples things:

1. **Mechanism parity.** Build a param-matched *transformer* baseline ourselves (C7) and compare
   *mechanisms* at equal size, equal data, equal budget. The question is "does complex/pattern beat
   attention **at the same scale**?", not "does it beat GPT?".
2. **The Pareto curve.** Plot quality vs cost (params, FLOPs, memory, latency, energy) for both
   architectures across several small sizes. The claim we can actually support is *"on the
   quality-vs-cost frontier, at small scale, our model sits here."*
3. **Scaling extrapolation.** Fit the small-scale scaling trend for both and **extrapolate**,
   clearly labeled as extrapolation with confidence intervals — never presented as a measured
   large-scale result. This is how you say something honest about scale without running at scale.

```
   quality
     ^          transformer baseline ●───●───●
     |                         ●───●
     |   ours ○───○───○───○
     |     ○──○
     +-------------------------------------> cost (params / FLOPs / energy)
        (compare frontiers; extrapolate dashed; never claim a GPT head-to-head)
```

---

## Instrumentation (what we log on every run)

- **Collapse:** effective rank (participation ratio of singular values) and per-dimension variance
  of predicted patterns, per epoch (Cut P).
- **Phase health:** histogram of learned phases (is phase actually being used, or stuck at 0?).
- **Norm drift:** `⟨ψ|ψ⟩` over depth (is unitary/renorm behaving?).
- **Brightness:** per-concept counts `n_c` and brightness `π_c` (Cut with glow).
- **Calibration:** reliability diagram + ECE (glow vs no-glow).
- **Cost:** wall-clock, peak RAM, params, FLOPs/step — for the Pareto curve.

---

## Deliverable of the prototype

A single reproducible report with: (1) Cut P G0 pass/fail; (2) Cut A accuracy + phase=0 ablation
gap (H1); (3) Cut B distance-to-parity and which mitigation helped; (4) the Pareto curve vs our own
param-matched transformer; (5) all configs + SHAs for bit-for-bit reproduction. Adjectives are
banned from the report; only numbers with error bars.

---

### Interview questions this doc answers

- *"How would you compare this to GPT?"* We wouldn't head-to-head. We compare **mechanisms at equal
  scale** against a transformer we built ourselves, plot the **Pareto curve**, and **extrapolate**
  the scaling trend — labeled as extrapolation.
- *"What stops you from cheating on the comparison?"* Equal params (±5%), equal tuning budget, equal
  data/loop, one switch at a time, ≥5 seeds with σ, and bit-for-bit-reproducible configs.
- *"Which cut runs first and why?"* Cut P (pattern-as-token), because it is the most valuable and
  most killable leg, and its G0 gate is the project-level go/no-go.

### Operator's scar

A slide once read "our 12M-param model is within 8 points of a model 1000× its size." True, and
completely misleading — the audience heard "almost as good as GPT." We killed the slide. The rule:
*never put our number and a frontier-model number in the same visual without the cost axis*, because
a bare accuracy comparison across a 1000× size gap is a rhetorical trick, not a result — and the
Pareto plot above exists precisely so the cost axis is never optional.
