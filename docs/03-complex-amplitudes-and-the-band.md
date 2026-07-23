# 03 — Complex amplitudes, and the "not 0 and 1, a band" question

A recurring framing is: *"instead of a bit that's 0 or 1, use a **band** of values."* This doc
answers it honestly. The framing is **half right and half a trap**, and getting the half-truth
straight is essential to knowing what is actually novel here.

---

## The half that's a trap: neural nets are already a "band"

A digital *bit* is `{0, 1}`. But a neural network **activation** has been a continuous real number
in a range (a "band") since the 1950s. Every weight, every activation, every embedding coordinate
in GPT is already an `f32`/`bf16` — a value in a continuous band, not a bit. So "use a band instead
of 0/1" describes **what neural nets have always done**. If you pitch that as the novelty, a
knowledgeable listener will (correctly) hear "we reinvented floating point," and you lose the room.

**The correction:** the axis you're adding is not *continuous magnitude* (already there). It is a
**second, orthogonal degree of freedom: phase.**

---

## The half that's real: complex phase is a genuinely new axis

A **complex amplitude** is `α = r·e^{iφ}` — two numbers:

- `r ≥ 0` — **magnitude** (how strongly the concept is present). This is the "band" nets already have.
- `φ ∈ [0, 2π)` — **phase** (the concept's "flavor" / its relationship to context). This is new.

Phase is not "a bigger band." It is an **orthogonal** dimension: two amplitudes can have the *same
magnitude* and behave *oppositely* because of phase. A real-valued net has no native place to put
this — it must spend extra parameters and depth to emulate "same strength, opposite effect."

Geometrically: a real activation is a point on a line; a complex amplitude is a point in a plane
(or a vector on a circle at radius `r`). The extra dimension is the circle's angle.

---

## Interference: where phase earns its keep

Add two amplitudes and take the Born-rule probability (squared magnitude):

```
|ψ₁ + ψ₂|² = |r₁e^{iφ₁} + r₂e^{iφ₂}|²
           = r₁² + r₂² + 2·r₁·r₂·cos(φ₁ − φ₂)
                          └────────── interference term ──────────┘
```

The **interference term** `2r₁r₂cos(φ₁−φ₂)` depends **only on the relative phase** `Δφ = φ₁−φ₂`:

| Relative phase `Δφ` | `cos(Δφ)` | Effect | Meaning |
|---|---|---|---|
| `0` (in phase) | `+1` | **Constructive**: `\|ψ₁+ψ₂\|² = (r₁+r₂)²` | Reinforcement (agreement) |
| `π/2` | `0` | **None**: `r₁²+r₂²` | Independent (classical add) |
| `π` (out of phase) | `−1` | **Destructive**: `\|ψ₁+ψ₂\|² = (r₁−r₂)²` | Cancellation (e.g. negation) |

Two equally strong concepts can **reinforce** or **cancel** based purely on their relative phase.
This is the single most important sentence in the doc: *phase gives you cancellation for free.*
In language, "not X" wants to *cancel* X; a complex model can represent that as a phase relationship
rather than learning a bespoke nonlinear gate. Whether this actually pays off at equal parameters is
**H1** (doc 00) — the decisive ablation.

---

## The Born rule: turning amplitudes into probabilities

To get an actual probability out of amplitudes, square the magnitude and normalize:

```
p_k = |α_k|²  ,   with   Σ_k |α_k|² = 1        (the state is normalized: ⟨ψ|ψ⟩ = 1)
```

This is the **Born rule**. It is why the state vector is "normalized to unit length" and why
readout is *squared magnitude*, not the raw value. It also means the representation lives on a
**unit sphere** in complex space, which pairs naturally with **unitary** (sphere-preserving)
evolution (doc 04).

---

## The honest trade: exactness/energy vs robustness

Adding phase is not free. Two costs to state plainly:

1. **Exactness / energy.** Maintaining meaningful phase relationships requires the arithmetic to be
   done in complex numbers and (for the dynamics) to be roughly norm-preserving. That is a
   *tighter numerical regime* than sloppy real-valued nets tolerate — closer to "you must keep the
   books balanced" (unit norm) than "anything goes." In physics terms, phase coherence is the thing
   that decoheres first; in our classical sim it costs discipline (careful normalization, unitary
   parameterization) rather than qubits, but the discipline is real.
2. **Robustness.** In exchange, phase relationships can be **more robust to certain perturbations**:
   information stored in *relative* phase survives global rotations, and unitary evolution cannot
   blow up or vanish (norm is conserved). You trade "cheap and forgiving" real arithmetic for
   "disciplined but norm-stable, cancellation-capable" complex arithmetic.

The experiment's job is to find where that trade is *worth it* (H1). If it never is at equal
params, we keep the pattern/glow legs and drop the complex branding.

---

## How you actually compute this (no complex hardware required)

Every framework can do complex via **`(re, im)` real pairs**:

- Store a `d`-dim complex vector as two `d`-dim real vectors, `re` and `im`.
- Complex multiply `(a+bi)(c+di) = (ac−bd) + (ad+bc)i` → four real multiplies and two adds.
- **PyTorch** has native `torch.complex64/128` and Wirtinger-calculus autograd through them.
- **Burn / Candle (Rust)** have no first-class complex tensor, so we implement a small `Complex`
  newtype over `(re, im)` pairs — portable across the NdArray/CPU and Metal/WGPU backends, and
  autodiff-native because it is just real ops underneath (doc `implementation/00-...`).
- On Apple **Metal**, complex support is partial, so the `(re, im)`-pair route is also what keeps
  us backend-agnostic: correctness on CPU first, speed on Metal second (C3, C4).

Nothing here needs special hardware — it is real linear algebra with the channels bookkept as pairs.

---

### Interview questions this doc answers

- *"Isn't 'a band instead of a bit' just floating point?"* Yes — that half is a trap; nets are
  already continuous. The real novelty is **phase**, an orthogonal second axis, not a wider band.
- *"What does phase buy you?"* Native **interference**: `2r₁r₂cos(Δφ)` lets equal-strength concepts
  reinforce (Δφ=0) or cancel (Δφ=π) — cancellation (e.g. negation) for free.
- *"How do you compute complex numbers without special hardware?"* `(re, im)` real pairs; native in
  PyTorch, a small newtype in Burn/Candle, portable to Metal.

### Operator's scar

Our first whiteboard pitch said "we use a band, not a bit." A staff engineer replied, deadpan,
"you mean… a float?" and the credibility hit took twenty minutes to recover. Since then the phrase is
banned. We say exactly one thing: *the new axis is phase, and phase buys interference.* If phase
can't beat a phase=0 ablation at equal params (H1), we don't get to say "quantum" at all — and we've
pre-committed to that outcome so the branding can't outlive the evidence.
