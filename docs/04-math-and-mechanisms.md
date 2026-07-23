# 04 — The math and mechanisms (each equation with its cost term)

This is the implementable core. Every mechanism is written as an equation, paired with its
**noise/cost term** (what it costs to run, where it can go wrong), because an equation without its
cost is a wish. All of it is standard complex linear algebra, trained by ordinary backprop.

Notation: `ψ` is a `d`-dim complex state (`ψ ∈ ℂ^d`), stored as `(re, im)` pairs. `⊙` is
element-wise (Hadamard) product. `⟨a|b⟩ = Σ_k conj(a_k)·b_k` is the complex inner product.

---

## 0. Digital baseline (the thing we must beat fairly, C7)

```
y = φ(Wx + b)
```

A standard real-valued layer: linear map `Wx+b`, pointwise nonlinearity `φ`. Stacked, this is the
param-matched baseline every complex mechanism below is compared against. **Cost:** `O(d²)` per
layer, real arithmetic, no norm constraint. **Failure mode:** none novel — this is the control.

---

## 1. Complex embedding

```
ψ = r ⊙ e^{iφ}          (r ∈ ℝ_{≥0}^d  magnitudes,  φ ∈ ℝ^d  phases)
```

Each input symbol maps to a magnitude vector `r` and a phase vector `φ`; the state is their polar
combination. Implemented as two real embedding tables (or one table of `(re, im)`).

**Cost / noise term.** Two real parameters per coordinate instead of one → **2× embedding
parameters** for the same `d`. Phase is periodic (`φ` and `φ+2π` are identical), so the optimizer
sees a wrapped landscape; we either let phase float and wrap, or parameterize `e^{iφ}` directly as
a unit-norm `(re, im)` pair. **Failure mode:** if `r` collapses to a constant, phase carries no
weighted information (guard with the phase=0 ablation, H1).

---

## 2. Interference and modulation (how states combine)

**Interference (superposition / addition):**

```
ψ_sum = ψ₁ + ψ₂
|ψ_sum|² = r₁² + r₂² + 2·r₁·r₂·cos(φ₁ − φ₂)      ← the interference term
```

**Modulation (multiplicative binding):**

```
ψ_mod = ψ₁ ⊙ ψ₂ = (r₁ ⊙ r₂) · e^{i(φ₁ + φ₂)}
```

Interference **adds** amplitudes (phases can cancel); modulation **multiplies** magnitudes and
**adds** phases (a binding/gating operation). Wave Network (doc 01) uses both; interference gave
90.91% and modulation 91.66% single-layer on AG News.

**Cost / noise term.** Both are `O(d)` element-wise — cheap. The subtlety is **numerical**: after
many interference steps the norm drifts, so we renormalize (Born rule wants `⟨ψ|ψ⟩=1`).
Renormalization is `O(d)` but must be logged (C4) because it silently changes gradients if applied
inconsistently between train and eval. **Failure mode:** unnormalized accumulation → magnitude
blow-up or vanishing → dead phase.

---

## 3. Unitary evolution (information-preserving dynamics)

We want a map `ψ' = Uψ` that **preserves norm** (`U†U = I`, unitary). Three ways to parameterize a
learnable unitary:

**(a) Matrix exponential of a Hermitian generator:**

```
U = e^{iH} ,   H = H†  (Hermitian)
```

Any Hermitian `H` gives a unitary `U`; learn `H`, exponentiate. **Cost:** `expm` is `O(d³)`
(eigendecomposition), which is expensive for large `d` and awkward to autodiff stably.

**(b) Cayley transform of a skew-Hermitian matrix:**

```
U = (I − A)(I + A)^{-1} ,   A = −A†  (skew-Hermitian)
```

The Cayley map turns any skew-Hermitian `A` into a unitary `U` using a **matrix inverse** instead
of an exponential. **Cost:** one `O(d³)` solve, but numerically friendlier and cleanly
differentiable. Preferred on CPU / LibTorch. **Failure mode:** `(I+A)` near-singular (eigenvalue of
`A` near `−1` on the relevant axis) → ill-conditioned solve; mitigate by bounding `A`.

**(c) Products of Givens rotations / Householder reflections:**

```
U = G_1 · G_2 · … · G_m       (each G a 2-dim rotation or a reflection)
```

Compose many cheap, **inverse-free** unitary primitives. **Cost:** `O(d)` per rotation, `O(d²)` for
a full-connectivity layer, no matrix inverse — **Metal/GPU-friendly** (Phase 5). **Failure mode:**
limited connectivity per layer → need enough rotations for expressivity.

**The bridge to real arithmetic.** On stacked `(re, im)` vectors `[Re(ψ); Im(ψ)] ∈ ℝ^{2d}`, a
**unitary** on `ℂ^d` is exactly an **orthogonal** matrix on `ℝ^{2d}` with the special block
structure `[[A, −B],[B, A]]`. So "unitary complex evolution" = "structured orthogonal real
evolution" — implementable with real ops and real autodiff, no complex kernels required.

---

## 4. Born-rule readout (turning states into predictions)

**Classification** (small, fixed set of classes `c`):

```
p_c = |⟨M_c | ψ⟩|²          (M_c a learned "measurement" vector per class; normalize over c)
```

**Cost:** `O(|C|·d)`, cheap for small `|C|` (e.g. 4 AG News classes). This is the regime where
Wave Network / C-NNQLM already win (doc 01).

**Generation** (predict next symbol over vocabulary `V`):

```
p_k = |⟨e_k | O ψ⟩|²   ∀ k ∈ V        (O an output map, e_k the k-th basis/codeword)
```

**Cost / THE WALL.** This is `O(|V|·d)` **per step** — the same order as a softmax over a large
vocabulary, but with the added constraint that outputs are squared magnitudes summing to 1. For
`|V| = 32000, d = 512` that is ~16.4M multiply-adds per step *just for readout*, and the output
matrix is itself `|V|·d ≈ 16.4M` params — the **vocabulary table** that pattern-as-token wants to
reclaim (doc 07, doc 09). **This is the single biggest feasibility risk for generation** and the
reason Phase 6 is gated.

**Mitigations (each an ablation in Phase 6):**

| Mitigation | Mechanism | Saves |
|---|---|---|
| **Low-rank** output `O = UVᵀ` | factor the `\|V\|×d` map through rank `k` | `O(k(\|V\|+d))` vs `O(\|V\|·d)` |
| **Tied** embeddings | reuse input embedding table as output | halves the vocab params |
| **Hierarchical** softmax/readout | tree over `V` | `O(d·log\|V\|)` per step |
| **VQ codebook** | predict into a small codebook, decode later | readout over `\|codebook\| ≪ \|V\|` |

---

## 5. Training: plain complex backprop (no quantum overhead)

The whole model is trained by **ordinary gradient descent** using **Wirtinger calculus** (the
standard way to differentiate real-valued losses of complex variables):

```
∂L/∂conj(z)  drives the update;   for real loss L,  z ← z − η · ∂L/∂conj(z)
```

This is **native in PyTorch** (`torch.complex64` autograd) and implemented over `(re, im)` pairs in
Burn/Candle (where it is *literally* real-valued autodiff, since a complex op is just real ops).

**What we explicitly do NOT pay** (the quantum-hardware taxes we avoid by running classically):

- **No parameter-shift rule** (the quantum way to estimate gradients circuit-by-circuit).
- **No shot noise** — gradients are exact to float precision, not sampled from measurements.
- **No barren plateau** — gradients are ordinary, not exponentially small in system size (doc 02).

**Cost / noise term.** Complex ops are ~**2–4×** the FLOPs of the equivalent real ops (the `(re,im)`
bookkeeping), and unitary parameterizations add an `O(d³)` (Cayley) or `O(d²)` (Givens) term per
unitary layer. That is the price; it is polynomial, deterministic, and M1-affordable (doc 09).

---

## The mechanism ledger (equation → cost → switch)

| Mechanism | Equation | Cost | Ablation switch (C6) |
|---|---|---|---|
| Digital baseline | `y = φ(Wx+b)` | `O(d²)` | — (control) |
| Complex embedding | `ψ = r⊙e^{iφ}` | 2× embed params | `phase = 0` → real |
| Interference | `ψ₁+ψ₂`, `\|·\|²` term | `O(d)` + renorm | `no_interference` |
| Modulation | `ψ₁⊙ψ₂` | `O(d)` | `no_modulation` |
| Unitary evolution | `U=e^{iH}` / Cayley / Givens | `O(d³)`/`O(d²)` | `no_unitary` (→ free linear) |
| Born classify | `p_c=\|⟨M_c\|ψ⟩\|²` | `O(\|C\|·d)` | `softmax` (real) |
| Born generate | `p_k=\|⟨e_k\|Oψ⟩\|²` | **`O(\|V\|·d)`** | low-rank / tied / hier / VQ |

---

### Interview questions this doc answers

- *"Write the core equations."* Embedding `ψ=r⊙e^{iφ}`; interference `|ψ₁+ψ₂|²=r₁²+r₂²+2r₁r₂cos Δφ`;
  unitary `ψ'=Uψ` (Cayley/Givens); Born readout `p=|⟨M|ψ⟩|²`.
- *"What's the expensive part?"* Generation readout `O(|V|·d)` per step — the vocabulary wall —
  mitigated by low-rank/tied/hierarchical/VQ.
- *"How is it trained, and what do you avoid by being classical?"* Plain Wirtinger backprop (native
  in PyTorch, real-pair in Burn) — no parameter-shift, no shot noise, no barren plateau.

### Operator's scar

We spent a week debugging "exploding loss" in the unitary layer before realizing we were applying
Born renormalization at eval but not at train, so the eval state lived on the unit sphere and the
train state didn't — the gradients were being computed for a different geometry than we scored. The
scar is now a rule in the ledger: *every normalization is a logged, switchable mechanism (C4/C6),
applied identically in train and eval, or it is a bug waiting for a deadline.*
