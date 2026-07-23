# Operational test protocol — every gate as a runnable PASS/FAIL

This is the **testability contract**. A gate that cannot be computed to an unambiguous PASS/FAIL from
a logged run is not a gate. Every threshold below is a **pre-registered constant** (fixed before the
run, with a one-line justification), not a "~" target. Where the earlier drafts hedged ("~90%",
"within ~10%", "10× a constant"), this document replaces the hedge with an exact number and the
procedure to compute it. The phase docs and white-paper §9 defer to this file for definitions.

Principle: each metric is a pure function `metric(run_artifacts) -> float`, and each gate is
`gate(metric) -> {PASS, FAIL}` with the threshold and the statistical test fixed here.

---

## 0. Statistical protocol (applies to every comparison)

- **Seeds.** Every reported number is the mean over **S = 5** seeds (seeds `0..4`, logged). Report
  mean and sample standard deviation `σ`.
- **Paired comparisons.** When two arms differ by one switch (C6), pair them **by seed** and test the
  per-seed difference with a **one-sided paired t-test at α = 0.05** (Wilcoxon signed-rank as the
  non-parametric fallback; pre-registered which, per metric, below). "Significant" means `p < 0.05`
  on this paired test.
- **Effect size AND significance.** A comparison PASSES a "better than" claim only if it clears **both**
  a pre-registered minimum effect size `δ` **and** significance. This is what removes the "if and but":
  a 0.1% win that is statistically real is still a FAIL against the effect-size floor.
- **Pre-registration.** The thresholds in §6 are frozen in the run config (`gates.toml`) before the
  first real training step; the config's git SHA is logged with every number (C4). Changing a
  threshold after seeing results voids the run.

---

## 1. The comparable quality metric: bits-per-byte (BPB)

The earlier phrase "validation perplexity via a fixed decoder" was **not operational** — a
next-pattern predictor does not natively output a token distribution. We replace it with a single,
exactly-defined, tokenizer-free metric that both arms produce identically.

**Definition.** On a held-out byte stream `x_{1..N}` (bytes, C2), any model that emits a next-byte
distribution `p_t(b) = P(x_t = b \mid x_{<t})`, `b ∈ {0..255}`, has

```
BPB = − (1/N) · Σ_{t=1..N} log2 p_t(x_t)            (bits per byte; lower is better)
```

**Both arms emit `p_t`, apples-to-apples:**
- **Token baseline:** softmax over 256 bytes (its native head).
- **Pattern model:** a **256-way Born readout head** on the predicted pattern,
  `p_t(b) = |⟨m_b | ẑ_t⟩|² / Σ_{b'} |⟨m_{b'} | ẑ_t⟩|²` (learned measurement vectors `m_b`, doc 04).
  This head is `256·d ≈ 0.131`M params for `d = 512` — counted in the budget, negligible (§8.3 of the
  paper). The pattern model is trained with `L = L_pattern + anti-collapse + λ·L_byte-CE` (the readout
  head's cross-entropy); the head is what "measures to speak."

**Two BPB numbers are logged; the gate uses the first:**
1. **Deployable BPB** (gate): head trained jointly with the encoder. This is the number a shipped
   byte-LM would have.
2. **Linear-probe BPB** (diagnostic): encoder frozen, only the head trained (a linear probe on the
   representation). Measures how much byte-predictive information the *pattern* carries, independent of
   the head's capacity. Reported alongside, never gated on.

Why BPB and not "perplexity": BPB is tokenizer-independent (satisfies C2), directly comparable across
any two byte-level models, and has no free "decoder" hyperparameter. Token perplexity would depend on
a tokenizer, which C2 forbids as a default.

---

## 2. Collapse (H3) — non-vacuous, relative to the target

**Bug fixed here.** The earlier gate "per-dim variance ≥ **10× a constant predictor**" is **vacuous**:
a constant predictor has variance `0`, and `10 × 0 = 0`, so any non-zero variance passes. Collapse is
measured **relative to the target patterns** (the thing the predictor is trying to match), which sets
the achievable scale.

Let `Z ∈ ℝ^{N×d}` be predicted patterns over the val set and `Z* ∈ ℝ^{N×d}` the (stop-gradient)
target patterns for the same items. Singular values `σ_1 ≥ … ≥ σ_d`.

```
erank(Z)      = (Σ_i σ_i)² / (Σ_i σ_i²)                 # participation ratio, ∈ [1, d]
meanstd(Z)    = (1/d) Σ_j std(Z_{·,j})                  # mean per-dimension std
```

**Non-collapse PASS (both must hold):**

```
erank(Z)   ≥ 0.50 · erank(Z*)          # predictions span ≥ half the target's intrinsic dimensionality
meanstd(Z) ≥ 0.50 · meanstd(Z*)        # predictions are no more than 2× "flatter" than targets, per dim
```

Justification of `0.50`: a predictor recovering less than half the target manifold's effective
dimensionality (or shrinking per-dim spread below half) has discarded most of the representation — the
operational definition of partial collapse. Pre-registered; reported as the raw ratios so the margin
is visible.

**Sanity gate (must run first).** A deliberately-collapsed toy (constant encoder) must read
`erank ≈ 1` and `meanstd ≈ 0`; if the metric does not scream on the toy, the metric is wrong (PHASE-1
order-of-operations).

---

## 3. Feasibility / competitiveness (H2, the G0 gate)

```
G0 competitive  ⟺  BPB_deploy(pattern) ≤ 1.10 · BPB_deploy(token baseline)      (§1)
                   AND non-collapse (§2) holds
                   over S = 5 seeds (mean); the 1.10 factor is the pre-registered "competitive" bar.
```

The `1.10` (a 10% BPB budget) is the "within small gap at fewer params" bar; it is a **feasibility**
threshold, not an improvement claim. FAIL ⇒ **STOP the project** (the pattern objective is load-bearing).

**The clean improvement claim (no if-and-but), reported as a Pareto point.** Feasibility (≤ 1.10×) is
only half the story; the *improvement* is stated in two exact, unconditional forms:
1. **Iso-`(d,L)` parameter efficiency (arithmetic certainty).** At equal width and depth the pattern
   model has **2.28× fewer parameters** than the token baseline (§8.3; verified by the param counter,
   not by training). This is true by construction — no training, no if/but.
2. **Iso-parameter quality (the head-to-head improvement).** Give the pattern model the *reclaimed*
   parameter budget back (extra depth/width until params match the token baseline to ±5%). The
   improvement gate is then:
   ```
   BPB_deploy(pattern @ equal params) < BPB_deploy(token) − δ_bpb,  δ_bpb = 0.02 bits/byte, significant.
   ```
   This is the one unconditional "it is better" claim: same parameter count, lower BPB by ≥ 0.02
   bits/byte, paired over 5 seeds, `p < 0.05`. If it does not clear this, we do **not** claim a quality
   improvement — we claim only the iso-`(d,L)` efficiency fact (1), which stands regardless.

---

## 4. Encoder invariance (H4) — the headline claim, made testable and non-gameable

Two properties were previously conflated and must be separated:

- **(4a) Attractor stability** (the memory mechanism, Phase 2): denoise a *code*.
  `retrieve(corrupt(ξ_c)) ≈ ξ_c` in code space.
- **(4b) Encoder invariance** (the **headline**, H4): the *encoder* maps *augmented inputs* to the same
  pattern. `enc(aug(x)) ≈ enc(x)` in pattern space. This is the property the model is named for, and it
  is what the reframed thesis claims.

Both are defined below with exact numbers. **(4b) is trained, not merely hoped for** — see §4.4.

### 4.1 The C1-clean augmentation family `A`

Augmentations must need **no pretrained model and no human labels** (C1). The pre-registered family,
applied to a byte string `x`:

| aug | operation | rates (pre-registered) |
|---|---|---|
| `delete` | drop each byte i.i.d. | `p ∈ {5%, 10%, 20%}` |
| `substitute` | replace each byte with a uniform random byte i.i.d. | `p ∈ {5%, 10%, 20%}` |
| `case` | flip alphabetic case per letter | `p = 50%` |
| `space` | insert/delete a space at word boundaries | `±1` per boundary |
| `swap` | swap adjacent bytes within a window | window `w = 3` |
| `code-noise` | Gaussian on the *code* (for 4a only) | `σ = 0.10 · meanstd(ξ)` |

"Paraphrase / synonymy" invariance (true semantic equivalence) is **explicitly excluded** from the
headline because it cannot be generated C1-clean (it needs a paraphraser or human labels); it is a
**separate, later** test on a held-out human-labeled or quarantined set (§4.5), never the headline.

### 4.2 Hit definition

For sparse `k`-of-`N` codes use active-set Jaccard; for dense patterns use cosine. A **hit** is:

```
sim(enc(aug(x)), enc(x)) ≥ τ_hit,        τ_hit = 0.90 (cosine)  or  Jaccard ≥ 0.90 (sparse)
hit-rate(A, rate) = fraction of held-out items that hit under augmentation A at the given rate
```

### 4.3 The joint invariance + discrimination gate (anti-collapse)

Invariance **alone is gameable**: a collapsed encoder maps everything to one pattern and scores
perfect "invariance." So the gate requires **discrimination** too, via a **margin**:

```
within(A,rate)  = mean_x  sim(enc(aug(x)), enc(x))          # same item, augmented
between         = mean_{x≠y} sim(enc(x), enc(y))            # different items (held-out pairs)

PASS (H4)  ⟺  within(A, rate) ≥ 0.90
              AND between ≤ 0.30
              AND margin = within − between ≥ 0.50
              at the pre-registered rates {delete 10%, substitute 10%, case 50%, swap w=3},
              over S = 5 seeds (mean).
```

Justification: random near-orthogonal codes have `between ≈ 0` (JL, doc 08), so `≤ 0.30` is generous
headroom; `within ≥ 0.90` is strict same-concept identity; `margin ≥ 0.50` guarantees the two are well
separated, which a collapsed model (`within ≈ between ≈ 1`, margin `≈ 0`) cannot fake. Report the full
`within`/`between` curve over all rates; the gate is evaluated at the pre-registered rates only.

FAIL ⇒ narrow the headline from "encoder invariance" to "attractor stability" (4a only), which is a
weaker but still-honest claim.

### 4.4 Invariance must be TRAINED, or following the plan will not achieve it

Next-pattern temporal prediction (`L_pattern`) does **not** by itself create augmentation invariance —
nothing in it tells the encoder that `aug(x)` and `x` mean the same thing. To *achieve* the headline
(not merely test it), the training objective includes an explicit invariance term:

```
L_inv = D(enc(aug(x)), stopgrad(enc(x)))       # augmented-view positive, JEPA/SimCLR-style
L      = L_pattern + L_inv + anti-collapse(§2) + λ·L_byte-CE(§1)
```

`L_inv` is a switch (C6): the ablation `no_invariance` (train without `L_inv`) must **fail** the §4.3
gate, demonstrating that the invariance is *caused by the objective*, not incidental. This closes the
"would this be achieved by following the files?" gap: the property is in the loss, and its necessity is
proven by ablation.

### 4.5 Semantic invariance (later, honestly scoped)

True synonymy/paraphrase invariance is measured on a **held-out labeled** set (human-labeled pairs, or
a clearly quarantined generator under the C1 quarantine protocol), reported separately, and never
folded into the headline number. It is a destination, like cross-modal invariance (§10 of the paper).

---

## 5. Calibration (H5)

The earlier gate `ECE(glow) < ECE(no-glow)` was under-specified (ECE depends on the bin count; a
`0.001` win "passes"; calibration can be bought by lowering accuracy). Pinned:

```
ECE (M = 15 equal-width confidence bins, held-out split only, C5):
  ECE = Σ_{m=1..15} (|B_m|/n) · | acc(B_m) − conf(B_m) |

PASS (H5)  ⟺  ECE(glow) ≤ ECE(no-glow) − δ_cal,   δ_cal = 0.02  (2 absolute ECE points)
              AND accuracy(glow) ≥ accuracy(no-glow) − 0.5 pt        # calibration not bought by dumbing down
              AND paired over S = 5 seeds, one-sided p < 0.05.
```

`M = 15` follows standard practice (Guo et al. 2017). The accuracy guard is what stops the degenerate
"be less confident about everything" solution from passing. FAIL ⇒ narrow "calibrated confidence" to
"salience weighting."

---

## 6. Phase benefit (H1) — the decisive ablation, with an effect size

```
Δ = metric(full-complex) − metric(phase=0),  param-matched (±5%), equal tuning budget (C6/C7).
  metric = AG News test accuracy (Cut A)  and  −BPB (Cut B), reported separately.

PASS (phase earns its parameters)  ⟺  Δ ≥ δ_phase  AND  paired one-sided p < 0.05 over S = 5 seeds
  δ_phase = 1.0 accuracy point (AG News)   or   ≥ 2% relative BPB improvement (byte-LM).
DROP the phase/"wave" claim              ⟸  Δ < δ_phase  OR  not significant
                                              OR the free-phase model drives φ → 0 on its own
                                              (phase-histogram diagnostic; the data did not want phase).
```

`Δ > 1σ` from the earlier draft is subsumed: the paired significance test *is* the "bigger than seed
noise" criterion, and `δ_phase` adds the effect-size floor so a real-but-trivial win still drops the
claim.

---

## 7. Pre-registration table (freeze before the first run)

| gate | metric | threshold (pre-registered) | consequence of FAIL |
|---|---|---|---|
| **G-stack** | oracle + finite-diff grad check | max rel. error `< 1e-4` per kernel | fix kernel; no downstream number is valid |
| **H3 collapse** | `erank(Z)`, `meanstd(Z)` | `≥ 0.50 · target`, both | STOP project |
| **G0 / H2** | deployable BPB ratio | `≤ 1.10 ×` token baseline | STOP project |
| **improvement (iso-`d,L`)** | param count | `2.28×` fewer (arithmetic) | — (certainty) |
| **improvement (iso-param)** | BPB gap | `< −0.02` bits/byte, `p<0.05` | claim only efficiency, not quality |
| **H4 invariance** | within / between / margin | `≥0.90 / ≤0.30 / ≥0.50` at pre-reg rates | narrow to attractor-stability |
| **H4 causation** | `no_invariance` ablation | must FAIL the H4 gate | invariance not attributable to `L_inv` |
| **H5 calibration** | ECE (M=15) + accuracy guard | `ΔECE ≤ −0.02`, acc `≥ −0.5pt`, `p<0.05` | narrow to salience |
| **H1 phase** | `Δ` accuracy / BPB | `≥ 1.0 pt` or `≥2%` rel, `p<0.05` | drop the phase/"wave" claim |

Every row is a function of logged artifacts with a fixed constant and (where it is a comparison) a
fixed statistical test. There is no undefined quantity and no "~".

---

### Interview questions this doc answers

- *"Your gate said 'perplexity via a fixed decoder' — how do you actually compute it?"* You don't; that
  was under-specified. Both arms emit a next-byte distribution and are scored by **bits-per-byte**; the
  pattern model gets a 256-way Born head for this. BPB is exact and tokenizer-free.
- *"'Variance ≥ 10× a constant predictor' — a constant has zero variance."* Correct, that was vacuous;
  fixed. Collapse is measured **relative to the target patterns**: `erank` and per-dim std each ≥ 0.50×
  the target's.
- *"How is 'invariance' not trivially won by mapping everything to one point?"* The gate requires a
  **margin**: `within ≥ 0.90`, `between ≤ 0.30`, `within − between ≥ 0.50`. A collapsed encoder has
  margin ≈ 0 and fails. And invariance is *trained* (`L_inv`) with an ablation proving causation.

### Operator's scar

The first gate sheet read "perplexity within ~10%, hit-rate ~90%, variance ≥ 10× a constant." Three
numbers, three holes: perplexity of a pattern model was undefined, "~" meant nobody had pre-registered
the bar, and "10× a constant" was `10 × 0 = 0` — a gate that everything passes. A reviewer asked "what
is the exact number and how do you compute it," and we could not answer for any of the three. The scar,
now a rule: **a threshold you cannot compute from a logged file, or that contains a "~", is not a gate —
it is a wish.** This document is where the wishes were turned into functions.
