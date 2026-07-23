# Phase 4 — Complex wave (the decisive H1 ablation)

**Goal:** determine whether **complex phase** is a real representational advantage or just extra
parameters. This is the ablation the whole "quantum" branding rides on (H1, docs 00/05). Run it once
the `Complex` stack is gradient-checked and trusted (Phase 0), because a wrong complex kernel would
fake either outcome.

Depends on: Phase 0 (`Complex` kernels) and Phase 1 (a working objective to attach them to).
Blocks: Phase 5 (unitary builds on complex).

**Test suite (execute this):** [`tests/PHASE-4.md`](tests/PHASE-4.md) — the param-matched `phase=0`
ablation (`nc_param_match` kills the rigged delete-imaginary mutant) and the H1 effect-size + significance verdict.

---

## What we build

1. **Complex embedding** `ψ = r ⊙ e^{iφ}` (two real tables, doc 04).
2. **Interference** `ψ₁ + ψ₂` with the Born term `|·|² = r₁²+r₂²+2r₁r₂cos Δφ`, and **modulation**
   `ψ₁ ⊙ ψ₂` (doc 04). Both switchable.
3. **Born readout** for the task (classification: `p_c = |⟨M_c|ψ⟩|²`).
4. The **decisive ablation switch**: `phase = 0`, which forces every amplitude real (`φ ≡ 0`),
   collapsing the model to a real-valued net **at the same parameter count** (the imaginary channel
   is zeroed, not removed, so params match — the honest way to isolate *phase*, not *param count*).

---

## The decisive comparison (H1)

Run, on the same task (Cut A / AG News is ideal — doc 06 — because the prior art says phase should
help there), param-matched, equal tuning, ≥5 seeds:

- **Full complex** (phase free).
- **phase = 0** ablation (real-valued, same params).

```
Δ = metric(full complex) − metric(phase=0)

PASS (phase earns its keep)  ⟺  Δ > 1σ of the seed noise   (complex clearly wins)
KILL (drop "quantum")        ⟸  Δ ≤ 1σ of the seed noise   (real ties → phase adds nothing)
```

This is the single most important experiment for the *branding*. If a phase=0 real net **ties** the
full-complex model at equal params and equal tuning, then **phase is not doing representational
work**, and we **drop the "quantum" claim** — keeping pattern-as-token (Phase 1) and glow (Phase 3),
which stand on their own. The project survives; the word "quantum" does not.

---

## Why this ablation is honest (and how it's usually rigged)

- **Param-matched, not mechanism-removed.** We zero the phase (`φ≡0`), keeping the parameter budget
  identical, so the comparison isolates *the phase degree of freedom*, not *net size*. Removing the
  imaginary tables entirely would confound phase with parameter count — the classic rigged ablation.
- **Equal tuning budget.** The real ablation gets the *same* hyperparameter search as the complex
  model. A common cheat is to tune the complex model hard and the ablation lazily; C6/doc-06
  fairness forbids it.
- **Seed distribution, not a point.** `Δ` is judged against the seed spread; a sub-σ "win" is noise.
  Pre-registering the σ threshold stops us from moving the goalposts after seeing the result.

---

## Supporting instrumentation

- **Phase histogram** over training: is phase actually being *used* (spread out) or is the optimizer
  driving it to 0 on its own? If the free-phase model *learns* `φ≈0`, that is itself an H1 kill —
  the data doesn't want phase.
- **Interference-term magnitude:** how large is `2r₁r₂cos Δφ` relative to `r₁²+r₂²`? If tiny, phase
  is contributing little even when nonzero.
- **Norm drift** after interference (renormalization sanity, doc 04 scar).

---

## The gate

```
KEEP "quantum" (proceed to Phase 5)   ⟺  Δ > 1σ AND phase is used (histogram non-degenerate)
DROP "quantum" (keep pattern + glow)  ⟸  Δ ≤ 1σ OR the free model learns φ≈0 on its own
```

Either way the project continues; only the branding is at stake. This is the cleanest kill in the
ladder because it is one number against one threshold, pre-registered.

---

### Interview questions this doc answers

- *"How do you know the 'quantum' part isn't just marketing?"* One pre-registered ablation: a
  phase=0 real net at **equal params and tuning**. If it ties (Δ ≤ 1σ), we drop "quantum" and keep
  the legs that stand alone.
- *"How do you avoid rigging the ablation?"* Zero the phase rather than deleting the imaginary
  channel (param-matched), give the ablation equal tuning, and judge Δ against the seed σ,
  pre-registered.
- *"What if the model just learns phase=0 by itself?"* That's an H1 kill too — the data didn't want
  phase — caught by the phase histogram.

### Operator's scar

Our first "complex beats real by 4 points" result evaporated when someone gave the real baseline the
same learning-rate sweep the complex model had gotten — the real net caught up to within noise. The
4 points were a tuning-budget artifact, not phase. The scar hardened Phase 4's rule: *the ablation
gets the identical search budget, and the σ threshold is written down before the run*, because the
easiest number to fake in this whole project is "complex beats real."
