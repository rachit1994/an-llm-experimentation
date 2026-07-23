# Phase 3 — Glow / confidence

**Goal:** turn exposure into a **calibrated, inspectable confidence** signal (C4, doc 10) and show
it beats a no-glow model on calibration. This is the product leg. Cheap to build once memory exists.

Depends on: Phase 2 (glow deepens the attractors that Phase 2 stores).
Blocks: nothing hard-required, but it is the on-device value story (doc 05 stakeholder framing).

---

## What we build (the mechanism from doc 10)

On each encounter of concept `c` (pattern `ξ_c`):

```
W  += η · ξ_c · ξ_cᵀ                       # Hebbian well-deepening
n_c += 1                                   # count
π_c  = normalize( log(1 + n_c) · idf(c) )  # brightness: log-compress × IDF salience, normalized
p(c) ∝ π_c   ⇔   |α_c| ∝ √π_c              # Born-consistent prior (probability = |amplitude|²)
confidence = π_top1 − π_top2               # retrieval margin
```

All three runaway fixes are **on by default and mandatory** (doc 10): **log-compression**,
**divisive/homeostatic normalization**, and **IDF-like salience**. They are individually switchable
(C6) *only* so we can demonstrate the runaway when they're off — never shipped off.

---

## Tests (three, each with a pass condition)

### Test 1 — Robustness rises with exposure

Retrain/expose concepts at varying frequencies; measure bee-test robustness (Phase 2) as a function
of `n_c`. **Pass:** robustness increases monotonically with exposure (deeper well = more noise
tolerance). This is the "glow = deeper attractor" claim, measured.

### Test 2 — Calibration beats no-glow (the H5 kill test)

Build a **reliability diagram** (predicted confidence vs empirical accuracy, binned) and compute
**ECE** (Expected Calibration Error) for the glow model and a no-glow model, matched otherwise (C6).

```
PASS   ⟺   ECE(glow) < ECE(no-glow)     (brightness tracks correctness)
NARROW ⟹   ECE(glow) ≥ ECE(no-glow)     (brightness ⟂ correctness)
           → drop "calibrated confidence / anti-hallucination",
             keep "salience weighting" (brightness as an attention prior only)
```

This is H5. It is the whole product claim reduced to one number: if the bright concepts are not the
*correct* ones, the confidence signal is a lie and we say so.

### Test 3 — Runaway check (mandatory, a guardrail not a metric)

Turn the three fixes **off** and confirm the model degenerates to "the brightest concept is the most
frequent token (e.g. the space / 'the')" — then confirm turning them **on** removes it. This proves
the fixes are load-bearing, not decoration. **Fail-safe:** if the shipped (fixes-on) model still
shows a frequency-dominated brightness distribution, that is a **bug**, not a result — do not report
calibration numbers until it's fixed.

---

## The gate

```
PROCEED (full glow claim)  ⟺  Test 1 monotone AND Test 2 ECE(glow) < ECE(no-glow) AND Test 3 clean
NARROW to salience         ⟸  Test 2 fails (ECE not beaten)
BUG (stop and fix)         ⟸  Test 3 shows frequency runaway with fixes ON
```

---

## Instrumentation

- Per-concept `n_c`, `π_c`, and the top-1/top-2 margin for every emission (inspectable, C8).
- Reliability diagram + ECE, glow vs no-glow.
- Brightness distribution (histogram) — must **not** be dominated by the highest-frequency tokens.
- Robustness-vs-exposure curve (Test 1).

---

## Why this is the leg a user actually touches

The abstain rule is one printable inequality: **emit if margin > τ, else hedge/abstain.** For an
on-device model that cannot defer to a bigger cloud model, "it tells you when it's guessing" is the
benefit (doc 05). Calibration (Test 2) is what makes `τ` a *meaningful* threshold instead of a knob;
without it, the margin is just a number.

---

### Interview questions this doc answers

- *"How do you prove the confidence is real, not vibes?"* Test 2: the glow model must have lower ECE
  than a matched no-glow model on a reliability diagram; if not, we narrow the claim to salience.
- *"What stops the most frequent token from being 'most confident'?"* Three mandatory fixes
  (log-compress, homeostatic normalization, IDF salience), and Test 3 explicitly checks the runaway
  is gone with fixes on and present with fixes off.
- *"What does the user get?"* An abstain rule: emit if the top1−top2 margin exceeds τ, else hedge —
  a built-in "I don't know" for an on-device model.

### Operator's scar

We reported a beautiful ECE improvement once — computed on the **training** distribution, where the
frequency prior trivially "calibrates" itself. On held-out text it was worse than no-glow. The scar:
*calibration is only meaningful on the held-out test split (C5)*, and Test 2 now runs exclusively on
test data with the splits hashed, because a confidence signal that only calibrates on data it has
seen is the exact opposite of anti-hallucination.
