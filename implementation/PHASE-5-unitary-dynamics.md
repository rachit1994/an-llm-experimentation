# Phase 5 — Unitary dynamics

**Goal:** determine whether an **information-preserving unitary** evolution layer buys real
long-range memory (the uRNN result, doc 01), or whether it is elegant dead weight. Drop it if no
gain.

Depends on: Phase 4 (unitary evolution acts on the complex state).
Blocks: Phase 6 (generation may use the recurrence).

---

## What we build (two parameterizations, chosen by backend)

A learnable **unitary** layer `ψ' = Uψ`, `U†U = I` (doc 04). Two routes:

1. **Cayley transform** — `U = (I − A)(I + A)⁻¹`, `A` skew-Hermitian.
   - Needs a **matrix inverse/solve** (`O(d³)`), numerically friendly, cleanly differentiable.
   - **Preferred on CPU / (quarantined) LibTorch**, where a stable solve is available.
2. **Givens rotations / Householder reflections** — `U = G₁·G₂·…·Gₘ`, each a cheap 2-D rotation or
   reflection.
   - **Inverse-free** (`O(d)` per primitive) → **Metal-friendly** (doc 09's Metal caveat).
   - Preferred on M1 GPU where a matrix inverse is awkward.

Both keep `U` exactly unitary by construction (no soft "orthogonality penalty"), so norm is
conserved and gradients neither vanish nor explode across many steps — the uRNN mechanism.

Reminder from doc 04: on stacked `(re, im)` vectors, a unitary on `ℂ^d` is a **structured orthogonal**
matrix on `ℝ^{2d}`, so both routes are implemented in real ops with real autodiff (Phase 0
gradient-checked).

---

## The probe: copy / long-range recall

Use the canonical **copy task** (and a recall variant) that LSTMs fail and uRNNs solve (Arjovsky et
al., doc 01): present a sequence, a long delay of blanks, then a recall cue; the model must reproduce
the early sequence after the delay. Sweep the delay length.

```
metric = recall accuracy vs delay length,   unitary layer  vs  a free (non-unitary) linear recurrence

PASS (keep unitary)  ⟺  unitary sustains accuracy at delays where the free recurrence collapses
DROP unitary         ⟸  no measurable long-range advantage over the free recurrence at equal params
```

The `no_unitary` switch (C6) replaces `U` with an unconstrained linear map of the same size, so the
comparison isolates *unitarity*, not *having a recurrence*.

---

## The gate

```
KEEP unitary   ⟺  clear long-range gain (accuracy holds at delays the free recurrence can't)
DROP unitary   ⟸  no gain → remove the layer; it's cost without benefit
```

Dropping unitary does not threaten the project or even the "quantum" branding (that rode on Phase 4's
phase result); it just means the *dynamics* leg didn't pay off and we ship without it.

---

## Cost / risk notes

- **Cayley conditioning:** `(I + A)` near-singular → ill-conditioned solve; bound `A`'s spectrum.
- **Givens expressivity:** too few rotations → limited connectivity; budget enough primitives for the
  effective mixing you need.
- **Nonlinearity caveat (doc 01):** a pure unitary map is linear and can't gate/forget; expect to
  interleave nonlinearity, which partially gives back the simplicity the formalism promised. Measure
  how much nonlinearity the probe actually needs rather than assuming.

---

### Interview questions this doc answers

- *"Why unitary instead of a normal linear layer?"* Norm preservation → gradients don't vanish/
  explode across long delays; it's the uRNN cure for long-range memory. Tested by the copy/recall
  probe against a free recurrence.
- *"How do you make it unitary on an M1 without a matrix inverse?"* Givens/Householder products
  (inverse-free, Metal-friendly); Cayley on CPU where a stable solve exists.
- *"When do you drop it?"* If it shows no long-range advantage over an equal-size free recurrence —
  it's then cost without benefit.

### Operator's scar

We implemented Cayley on Metal first because it was one clean formula, and spent days chasing NaNs
from an ill-conditioned inverse the GPU handled differently than the CPU. The Givens route — ugly,
many small rotations, but **inverse-free** — just worked on Metal. The scar: *match the unitary
parameterization to the backend* (Cayley on CPU, Givens on GPU), because "one elegant formula" is
worthless if the hardware can't invert the matrix stably.
