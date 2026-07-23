# implementation/00 — Stack and principles

The engineering charter for the build. It picks the stack, defines the complex-number
representation, sets the correctness discipline (a gradient check per kernel), and states the one
honest escape valve. Everything here serves [`../GROUND-UP-CONSTRAINTS.md`](../GROUND-UP-CONSTRAINTS.md).

---

## The Rust stack

- **Burn** — a backend-agnostic Rust deep-learning framework. Use the **NdArray (CPU)** backend for
  **correctness and bit-for-bit determinism** (C4), then the **Metal/WGPU** backend for **M1 speed**.
  Backend-agnosticism is the whole point: write the model once, run it on CPU (truth) and GPU
  (speed) without changing the code.
- **Candle** — a lighter Rust tensor library, kept as an alternative for the smallest kernels /
  fast iteration if Burn's abstraction is heavier than a given experiment needs.
- **No `tch`/libtorch, no CUDA runtime for headline numbers (C3).** LibTorch is allowed **only** in
  a clearly-quarantined perf comparison (e.g. "how much faster would a mature kernel be?"), never in
  a reported result. This keeps the M1 claim honest: an M1 number must come from an M1-runnable,
  pure-Rust binary.

Why Rust at all: determinism and control over numerics (C4), a single binary with pinned deps that
reproduces bit-for-bit, and no Python-environment drift. The cost is iteration speed, addressed by
the escape valve below.

---

## Complex numbers = a `(re, im)` pair newtype

Neither Burn nor Candle has a first-class complex tensor, and Apple **Metal has only partial complex
support** (doc 03/09). So complex is represented as **two real channels** behind a small newtype:

```rust
/// A complex value / tensor stored as (real, imaginary) parts.
/// Portable across NdArray(CPU) and Metal(WGPU); autodiff-native because
/// every method below is expressed in *real* tensor ops the backend already
/// differentiates.
struct Complex<B> { re: Tensor<B>, im: Tensor<B> }

impl<B: Backend> Complex<B> {
    // (a+bi)(c+di) = (ac − bd) + (ad + bc)i
    fn mul(&self, o: &Self) -> Self {
        Self { re: self.re.mul(o.re) - self.im.mul(o.im),
                im: self.re.mul(o.im) + self.im.mul(o.re) }
    }
    fn add(&self, o: &Self) -> Self { /* re+re, im+im */ }
    fn abs2(&self) -> Tensor<B> { self.re.powf(2.0) + self.im.powf(2.0) } // Born: |z|²
    // unitary-on-ℂ^d  ==  structured-orthogonal-on ℝ^{2d}  (doc 04)
}
```

Three properties this buys, all required by the charter:

1. **Portable** — it is just real tensors, so it runs on any Burn/Candle backend, including Metal
   (C3, doc 09's Metal caveat).
2. **Autodiff-native** — every operation is a real op the framework already differentiates; there is
   no custom complex-autograd to get wrong (this is Wirtinger calculus, made concrete — doc 04).
3. **Inspectable** — `re` and `im` are always readable, so the whole-state screenshot (C8, docs 02,
   10) is free.

---

## Repository layout

```
implementation/
  qilm-core/     # the model kernels: Complex, interference, modulation,
                 #   unitary (Cayley/Givens), Born readout, Hopfield retrieval
  qilm-train/    # training loops, objectives (L_pattern + anti-collapse), the
                 #   BINARY that produces headline numbers (pins its Cargo.lock, C4)
  qilm-data/     # byte/char loaders (C2 default), fixed hashed splits (C5),
                 #   the BPE loader behind an explicit `--tokenizer bpe` ablation flag
  qilm-oracle/   # scalar reference implementations + finite-difference gradient checks
```

(`qilm` = "quantum-inspired language model" — the working name; nothing about it presumes hardware.)

---

## Correctness discipline: a scalar oracle + finite-difference check per kernel

**Every kernel** in `qilm-core` ships with two things in `qilm-oracle`:

1. **A scalar oracle** — a dead-simple, obviously-correct reference implementation of the same math
   on plain `f64` scalars (no tensors, no backend). The tensor kernel must match the oracle to
   float tolerance on random inputs.
2. **A finite-difference gradient check** — verify the analytic gradient against
   `(L(θ+ε) − L(θ−ε)) / 2ε` for small `ε`, per parameter, on random inputs. If the kernel's gradient
   disagrees with finite differences, the kernel is wrong, full stop.

This is non-negotiable because **every headline number is a stack of these kernels**; one wrong
gradient silently poisons every result above it (the "train/eval normalization mismatch" scar,
doc 04). The gradient check is the cheapest insurance in the project.

---

## Determinism (C4, made operational)

- **Seed everything**: one master seed → per-component seeds, logged.
- **Pin every version**: the `qilm-train` binary force-adds its `Cargo.lock` (see `.gitignore`
  note); the toolchain is pinned via `rust-toolchain.toml`.
- **Log the full config with every number**: seed, versions, git SHA, shapes, hyperparameters,
  backend — emitted as a JSON sidecar next to every result.
- **CPU is the source of truth**: headline numbers reproduce **bit-for-bit on the NdArray backend**.
  Metal timings are recorded separately and re-verified on CPU; GPU reduction-order nondeterminism is
  never allowed into a headline number.

---

## The honest Rust-vs-iteration-speed tradeoff (and the escape valve)

Rust gives determinism and a clean M1 story but is **slower to iterate** than Python for
research-y, throwaway experimentation. We acknowledge this rather than pretend it away.

**Escape valve (Phase 1 ONLY, ≤ 1 week):** if Phase-1 feasibility (does the pattern objective even
train without collapsing?) is faster to answer in a **Python spike**, that is allowed — **time-boxed
to one week**, clearly labeled **quarantined** (it does not produce a headline number under C3), and
used only to **de-risk the idea before committing Rust effort**. The moment the spike answers "yes,
it trains," the real, reproducible result is re-implemented and re-run in Rust. The spike buys
*information*, never a *result*.

This escape valve exists for Phase 1 and nowhere else: once G0 is proven, the discipline (Rust, CPU,
pinned, gradient-checked) is mandatory for every reported number.

---

### Interview questions this doc answers

- *"Why Rust, and what's the cost?"* Determinism, control over numerics, and a clean pure-Rust M1
  story (C3/C4); the cost is iteration speed, bounded by a ≤1-week quarantined Python spike allowed
  *only* for Phase-1 feasibility.
- *"How do you do complex numbers with no complex tensor type?"* A `(re, im)`-pair `Complex` newtype
  over real tensors — portable to Metal, autodiff-native, inspectable.
- *"How do you know a kernel is correct?"* A scalar `f64` oracle plus a finite-difference gradient
  check per kernel; disagreement means the kernel is wrong.

### Operator's scar

We shipped a "fast" Metal kernel that gave a slightly different loss than the CPU version and spent
two days assuming the *CPU* was wrong. It wasn't — Metal's reduction order made the GPU number
nondeterministic, and we'd let it become the reference. The scar is the rule in bold: **CPU is the
source of truth**; the GPU is a stopwatch, never a witness.
