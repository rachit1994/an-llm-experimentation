# Prior-art reference implementations & the reuse/dependency policy

**Do not reinvent a solved problem, and do not drown the reviewer in dependencies.** Every primitive
in CCR already exists, implemented and tested, somewhere. Before you write a standard component, read
the reference row below, port its *algorithm, default hyperparameters, and test vectors*, and
implement it **minimally in-repo**. The rule is **"learn from the reference, do not depend on it."**

This doc is the engineering companion to the scientific bibliography
([`../docs/references.md`](../docs/references.md)) and the field positioning
([`../docs/01-prior-art-quantum-inspired.md`](../docs/01-prior-art-quantum-inspired.md) /
[`../docs/11-vsa-positioning-and-reframe.md`](../docs/11-vsa-positioning-and-reframe.md)). It is
binding via AGENTS.md rules 11–12.

---

## The honest framing (state it, don't hide it)

CCR is a **recombination** of mature, individually-solved ideas — VSA/HDC operations, complex-valued
layers, modern-Hopfield retrieval, joint-embedding (JEPA/VICReg) anti-collapse, Born-rule readout,
calibration/ECE, byte-level BPB. There is essentially **no primitive here that lacks a public
reference**. The contribution is the *specific intersection* (learned + from-scratch + wave-domain
binding + attractors + calibrated confidence) and the *verified, minimal-dependency engineering* — not
new building blocks. So the correct build strategy is: **port each primitive from its reference, spend
the effort on integration and the anti-fake verification, add almost nothing to `Cargo.toml`.**

---

## Dependency policy (minimal — a reviewer must survive `cargo tree`)

**Allowed core dependencies** (the whole budget; each justified, each a single small crate):

| crate | why | alternative if we drop it |
|---|---|---|
| `rand` + `rand_chacha` | seedable, reproducible RNG (C4 determinism) | none — determinism is non-negotiable |
| `serde` + `serde_json` | run artifacts / `metrics.json` / truth.json | hand-rolled JSON (avoid; error-prone) |
| `sha2` | provenance + split hashing (anti-fake) | hand-rolled hashing (don't) |
| `proptest` (**dev-only**) | property/metamorphic tests | hand-rolled generators |

**Deliberately NOT taken:**
- **No DL framework for headline numbers** — no `burn`, `candle`, `tch`, `dfdx`. `tch` is banned (C3,
  libtorch). `burn` is a large surface; `dfdx` needs nightly. If Phase-1 training needs reverse-mode
  autodiff, **hand-roll a ~300–500-line tape** (see "Rust stack" below) — dependency-free and
  reviewable (C8). `candle` is the *only* sanctioned framework fallback, and only with lead sign-off,
  as a single pure-Rust CPU dependency (never libtorch).
- **No FFT crate by default** — the binding-identity check (`kat_binding_identity`) runs at `d≈256`;
  hand-roll a small `O(d²)` DFT for tests (zero deps). Add `rustfft`/`realfft` **only** behind a
  `--features fft` flag if a real Phase-6 perf need is demonstrated, never for a headline correctness
  number.
- **No linear-algebra framework** — `nalgebra`/`ndarray` are not required for the small kernels; plain
  `Vec<f64>`/slices suffice and read clearly. Introduce `ndarray` (one crate) only if batched Phase-1
  training makes plain `Vec` genuinely unwieldy, and justify it in the PR.

**The rule for any new dependency:** justify it in the commit against this budget; prefer a small
hand-rolled implementation over a heavy crate; a DL framework needs lead sign-off. Keep `cargo tree`
shallow.

---

## Per-component reference map

For each: the canonical reference (repo + paper), **what to borrow**, the **build decision**, and any
**test vector** we can steal to strengthen our suite. All references are read-only; none become deps.

### VSA / HDC operations — bind, bundle (superpose), permute
- **Reference:** [`hyperdimensional-computing/torchhd`](https://github.com/hyperdimensional-computing/torchhd)
  (JMLR 2023, arXiv:2205.09208) — the standard VSA library; MAP/HRR/BSC models.
- **Borrow:** the exact operation semantics (bind = elementwise/Hadamard or circular convolution;
  bundle = sum; permute = fixed cyclic shift), and the **algebraic properties** as property tests:
  bind is invertible (unbind recovers), bind distributes over bundle, permute is invertible and
  bind/permute preserve near-orthogonality. These are free, high-value L1 tests.
- **Build:** implement in `qilm-core::bind` on `Vec<f64>` (a few lines each). **No dep.**

### Complex-valued layers + initialization
- **Reference:** *Deep Complex Networks* (Trabelsi et al., ICLR 2018, arXiv:1705.09792) +
  [`wavefrontshaping/complexPyTorch`](https://github.com/wavefrontshaping/complexPyTorch).
- **Borrow:** the `(re, im)` op algebra (already in doc 04); the **complex weight init** (magnitude
  ~ Rayleigh, phase ~ Uniform(−π, π)) so the phase model starts in a sane regime; complex-BN only if a
  run needs it.
- **Build:** `qilm-core::complex` as our `(re, im)` pair type (C8 inspectable). **No dep.**

### Modern-Hopfield retrieval
- **Reference:** [`ml-jku/hopfield-layers`](https://github.com/ml-jku/hopfield-layers) (official,
  arXiv:2008.02217).
- **Borrow:** the update `X·softmax(β XᵀΞ)`, `β = 1/√d`; the one-step-retrieval and exponential-capacity
  properties → the "bee test" thresholds (Phase 2) and a retrieval-error KAT.
- **Build:** `qilm-core::hopfield` (softmax + two matvecs). **No dep.**

### Anti-collapse objective (JEPA / VICReg / Barlow Twins)
- **References:** [`facebookresearch/vicreg`](https://github.com/facebookresearch/vicreg) (arXiv:2105.04906),
  [`facebookresearch/ijepa`](https://github.com/facebookresearch/ijepa) (EMA target),
  [`facebookresearch/barlowtwins`](https://github.com/facebookresearch/barlowtwins) (arXiv:2103.03230,
  the **fewer-hyperparameter** alternative I flagged for our ablation).
- **Borrow — exact defaults, so we don't guess:** VICReg `sim=25.0, std=25.0, cov=1.0`; variance is a
  hinge on per-dim std with target `γ=1`; covariance = sum of off-diagonal² divided by `d`; EMA target
  momentum from I-JEPA (~`0.996→1.0`). Barlow Twins: cross-correlation to identity, `λ≈5e-3`.
- **Build:** `qilm-train::loss` with the switches (`jepa_vicreg`, `barlow`, `infonce`, `vq`). **No dep.**
- **Steal a test vector:** compute VICReg loss on a fixed tiny batch by hand (or from the reference)
  and assert our implementation matches → a KAT, not a shape check.

### Unitary / orthogonal parametrization
- **Reference:** Lezcano-Casado *Cheap Orthogonal Constraints* (ICML 2019, arXiv:1901.08428),
  [`Lezcano/expRNN`](https://github.com/Lezcano/expRNN) — `orthogonal.py` covers exp-map, **Cayley**,
  and Householder.
- **Borrow:** the Cayley transform `U=(I−A)(I+A)⁻¹` for skew-symmetric `A` (our default, doc 04) and the
  Givens/Householder products (Metal-friendly). Their tests assert `UᵀU=I` — same as our `kat_unitary`.
- **Build:** `qilm-core::unitary` with a small linear solve for Cayley. **No dep** (no `nalgebra`; a
  small Gaussian-elimination solve suffices at our `d`).

### Calibration / ECE
- **Reference:** [`Lightning-AI/torchmetrics`](https://github.com/Lightning-AI/torchmetrics)
  `CalibrationError` (Guo et al. 2017).
- **Borrow — confirms our choice:** bin the top-1 confidence into **`n_bins=15` equal-width bins**,
  `ECE = Σ_b (|B_b|/n)·|acc_b − conf_b|` (L1). Also expose MCE (L∞) and RMSCE (L2) as diagnostics.
- **Build:** `qilm-train::metrics::ece` (trivial). **No dep.** Steal a fixed confidence/label set with a
  hand-computed ECE as the KAT (`test_ece_known`, PHASE-3 T3.2).

### Bits-per-byte (the quality metric)
- **Reference:** byte-level LM practice — nanoGPT (`karpathy/nanoGPT`), MambaByte (arXiv:2401.13660),
  SpaceByte (arXiv:2404.14408).
- **Borrow — confirms METRICS-AND-GATES §1:** `BPB = (mean per-byte NLL in nats)/ln 2` = log2
  cross-entropy per byte; tokenizer-agnostic (exactly why C2 uses it). No transformation needed since we
  are already byte-level.
- **Build:** `qilm-train::metrics::bpb`. **No dep.**

### Markov-source entropy (the known-answer fixture — already fixed)
- **Reference:** standard information theory; the Python `dit` library (`dit/dit`) is a cross-check for
  entropy rate, but **not a dep**.
- **Borrow:** entropy rate `H = Σ_i π_i · H(P_i·)` from the true matrix + stationary `π` — this is what
  the corrected `qilm-data` now does. Keep the anti-vacuity canary (AGENTS.md rule 2).

### Finite-difference gradient check
- **Reference:** `torch.autograd.gradcheck`, JAX `check_grads`, and the micrograd/autodiff-nd tape idea.
- **Borrow:** central difference `(L(θ+ε)−L(θ−ε))/2ε`, `ε=1e-4`, max relative error `< 1e-4`.
- **Build:** `qilm-oracle::gradcheck` (done). **No dep.**

### Reverse-mode autodiff for Phase 1 (the one hand-rolled engine)
- **References:** [`karpathy/micrograd`](https://github.com/karpathy/micrograd) (~100-line engine: a
  `Value{data, grad, _backward}` node, topological order, `.backward()` in reverse) and its Rust ports
  [`ferric-micrograd`](https://github.com/msminhas93/ferric-micrograd),
  [`danielway/micrograd-rs`](https://github.com/danielway/micrograd-rs), and the tape-style
  [`rustograd`](https://github.com/AnarchistHoneybun/rustograd).
- **Borrow the *architecture*, not the scalar granularity.** micrograd is scalar-valued — one node per
  add/mul — which is far too slow for a 12.6M-param model. Port its *idea* (per-node backward closure +
  topological reverse pass) but implement a **tensor-valued, index-based tape (Wengert list)**: a
  `Vec<Node>` arena where each `Node` holds a value buffer, a grad buffer, and the indices of its inputs;
  forward appends nodes, backward walks the arena in reverse accumulating grads. Index/arena beats
  `Rc<RefCell>` here (cache-friendly, no interior-mutability overhead) and is the standard fast pattern.
- **Ops needed for Phase 1 (small, enumerated):** `linear` (matmul + bias), an elementwise nonlinearity
  (`tanh`/`gelu`), `add`, a numerically-stable `log_softmax`/`cross_entropy` (for BPB), and the
  anti-collapse terms (VICReg variance/covariance, or InfoNCE). Nothing else.
- **Correctness gate (already built):** every op's backward is validated by `qilm-oracle::gradcheck`
  (finite-diff, max rel err `< 1e-4`) — so the hand-rolled tape is *proven*, not trusted. This is why we
  can hand-roll it safely instead of pulling a framework.
- **Build:** `qilm-core::autodiff` (a few hundred lines). **No dep** (no `candle`/`burn`/`dfdx`). `candle`
  remains the sign-off-only fallback if profiling proves the tape is the bottleneck (doc: dependency
  policy above).

---

## The Rust stack decision (minimal, phased)

- **Phase 0 (kernels + oracle + harness):** **zero** ML framework. Plain `Vec<f64>`/slices; finite-diff
  needs no autograd. Deps: only the core budget above.
- **Phase 1+ (training needs reverse-mode autodiff):** **hand-roll a small tape** — a `Var { value,
  grad }` graph with topological backprop, ~300–500 lines, modeled on karpathy/`micrograd` and
  `ArunBabu98/autodiff-nd`. Dependency-free, and its transparency *is* a feature (C8 inspectable). The
  models are small (12.6M params, doc 09), so a clean hand-rolled engine is adequate on CPU.
- **Only if profiling proves the hand-rolled engine is the bottleneck** for a headline run: escalate to
  the lead; the sole sanctioned fallback is [`huggingface/candle`](https://github.com/huggingface/candle)
  as a single pure-Rust CPU dependency. Never `burn`/`tch`/`dfdx` for headline numbers.

Rationale: a research reviewer should `git clone`, `cargo test`, and read the whole model without
learning a framework. Fewer deps = fewer version pins = reproducibility (C4) and no dependency hell.

---

### Interview questions this doc answers

- *"Isn't every piece of this already implemented elsewhere?"* Yes — and we say so. We port each
  primitive from its canonical reference (torchhd, hopfield-layers, VICReg, expRNN, torchmetrics),
  implement it minimally in-repo, and spend effort on the integration + anti-fake verification. The
  novelty is the intersection, not the primitives.
- *"Why not just use a Rust DL framework?"* Minimal-dependency reviewability. Phase 0 needs no
  framework; Phase 1 uses a ~400-line hand-rolled tape; `candle` is a sign-off-only fallback. A
  reviewer must survive `cargo tree`.
- *"How do you avoid reimplementing a standard algorithm wrong?"* Steal the reference's **test
  vectors** (VICReg loss on a fixed batch; ECE on a fixed set matching torchmetrics; bind/unbind
  properties from torchhd) and assert against them — a KAT, not a guess.

### Operator's scar

An agent burned a session re-deriving the modern-Hopfield update and the VICReg coefficients from the
papers, getting the covariance normalization subtly wrong — a bug the reference repo would have handed
us for free. The scar, now AGENTS.md rule 11: **read the reference and port its test vectors before you
implement; the goal is to spend our scarce effort on what is actually new (the integration and the
verification), not on re-deriving `std_coeff = 25.0`.**
