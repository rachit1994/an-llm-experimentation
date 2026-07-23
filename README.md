# an-llm-experimentation

A **ground-up** experiment in **learning stable, surface-invariant, inspectable concept
representations** from scratch. Not a fork of the transformer stack, not a wrapper around a
pretrained checkpoint, and not a quantum-hardware project. It builds a **learned Vector
Symbolic Architecture** — a distributed-pattern encoder whose *binding* operation is
implemented in the **wave / frequency domain** (complex amplitudes) — implemented in
**classical** linear algebra and sized to run on a **Mac Mini M1**.

The repo is a dossier first and a build ladder second. Every claim below is decomposed into
an individually falsifiable hypothesis with a *killable number*, and every mechanism is
switchable so that any reported result is attributable to one cause.

> **Framing note (v2).** An earlier version led with "Amplitude Language Model," which put the
> *mechanism* (complex/wave math) in the title as if it were the identity. The white paper now
> leads with the **problem** — a learned, from-scratch encoder that maps inputs to convergent,
> surface-invariant concept patterns — anchors it in **Vector Symbolic Architectures /
> Hyperdimensional Computing** (the field this idea actually belongs to; permutation and binding
> are *named VSA primitives*, not metaphors), and demotes the wave math to what it is: the
> **frequency-domain form of VSA binding** (`bind = circular convolution = elementwise complex
> multiplication = phase addition`). The math and every number are unchanged; only the framing is.

---

## One-paragraph verdict

The problem worth solving here is a **stable, surface-form-invariant, inspectable concept
representation**: the same concept — under paraphrase, augmentation, noise, and eventually a
different modality — should evoke nearly the same **distributed pattern**, that pattern should be
a reliable **attractor**, and the model should expose a **calibrated confidence**. The substrate
is **Vector Symbolic Architectures** (bind/superpose/permute over high-dimensional near-orthogonal
codes); the one mechanistic novelty is that **binding is done in the wave/frequency domain**, where
circular-convolution binding is exactly elementwise complex multiplication (phase addition) — so the
"amplitude" math is a *learnable realization of VSA binding*, not a separate thesis. The published
precedent for the mechanism is strong (Wave Network reaches **90.91%**/**91.66%** single-layer on AG
News vs a single transformer layer's ~**71–72%**, approaching BERT-base's **94.64%**). The **testable
headline** is single-modality invariance on an M1; **cross-modal** invariance and **generation at
scale** are stated destinations, not results (the latter gated behind the Born-readout `O(|V|·d)`
wall). A real quantum computer is the *wrong* tool here (barren plateaus, dequantization, the
simulability pincer), which is exactly why the whole thing runs classically. See the
[white paper](WHITEPAPER.pdf) and [`05-feasibility-verdict.md`](docs/05-feasibility-verdict.md).

---

## Theory → wave-domain mechanism → classical implementation

| Theory (plain language) | Quantum formalism | Classical implementation (M1) |
|---|---|---|
| A concept has a strength *and* a "flavor" | Complex amplitude `α = r·e^{iφ}` (magnitude `r`, phase `φ`) | A `(re, im)` float pair; `d`-dim complex vector as two `d`-dim real vectors |
| Concepts combine and can reinforce or cancel | Superposition + **interference** `\|ψ₁+ψ₂\|² = r₁²+r₂²+2r₁r₂cos(φ₁−φ₂)` | Complex add/multiply; modulation `ψ₁⊙ψ₂ = r₁r₂ e^{i(φ₁+φ₂)}` |
| Meaning evolves without losing information | **Unitary** evolution `ψ' = Uψ`, `U = e^{iH}` (`H` Hermitian) | Cayley `U=(I−A)(I+A)⁻¹` (CPU) or Givens/Householder rotations (Metal) |
| You must *measure* to get a word out | **Born rule** `p_k = \|⟨e_k\|Oψ⟩\|²`, `Σ p_k = 1` | Squared-magnitude readout head over classes/vocabulary |
| The unit of language is a *pattern*, not a symbol | State vector `\|ψ⟩` in a concept Hilbert space | Sparse distributed code; predict next **pattern** `ẑ_{n+1}`, decode to a word only at emission |
| Concepts are stable memories | Attractor / fixed point | Modern-Hopfield retrieval `X·softmax(β XᵀΞ)`, `β = 1/√d` |
| Familiar things feel brighter / more certain | Higher prior `p(c) ∝ π_c`, deeper well | Hebbian `W += η ξ_c ξ_cᵀ`; brightness `π_c = normalize(log(1+n_c))` |

---

## Executive verdict table

| Claim | What it asserts | Status | Kill number |
|---|---|---|---|
| **C1** Quantum *formalism* | Amplitudes + interference + unitary + Born, all classical | **PROCEED** | — (it's a framework, tested via C2–C4) |
| **C2** Pattern-as-token | Predict next *pattern*, decode at emission | **PROCEED (gated G0)** | Collapse OR >10% perplexity gap vs param-matched token baseline |
| **C3** Sparse-attractor capacity | Same concept re-triggers the same sparse nodes | **PROCEED** | "Bee test" hit-rate < ~90% under noise |
| **C4** Frequency "glow" | Calibrated, inspectable confidence from exposure | **PROCEED (narrow-able)** | Brightness uncorrelated with correctness (ECE not beaten) |
| Complex **phase** is the novelty | Real-valued (phase=0) ablation must *lose* | **DECISIVE ABLATION** | Real ablation ties at equal params ⇒ drop the quantum claim |
| Quantum **hardware** | A real QC would help | **REJECTED** | Barren plateaus / dequantization / simulability pincer |
| Generation **at scale** | Born readout over full vocab is affordable | **UNPROVEN** | Needs full `O(\|V\|·d)` with no mitigation ⇒ narrow to encoder-only |

---

## Reading order

0. **[`WHITEPAPER.pdf`](WHITEPAPER.pdf)** — the peer-review-grade research paper (theory, derivations, capacity bounds, all calculations, references), typeset with fully rendered math. **Read the PDF, not the `.md`, on mobile**: GitHub's math rendering (`$...$` / `$$...$$`) has never been supported in the iOS/Android apps, so equations show as raw LaTeX source there even though they render correctly on desktop/web. [`WHITEPAPER.md`](WHITEPAPER.md) is the Markdown source (renders with math on github.com desktop; see [`docs/PDF-BUILD.md`](docs/PDF-BUILD.md) to regenerate the PDF after editing it). Start here if you want the rigorous end-to-end argument; the docs below are the working dossier behind it.
1. [`11-vsa-positioning-and-reframe.md`](docs/11-vsa-positioning-and-reframe.md) — **the framing**: this is a *learned Vector Symbolic Architecture*; the wave math is frequency-domain VSA binding. Read early — it re-points everything below.
2. [`GROUND-UP-CONSTRAINTS.md`](docs/GROUND-UP-CONSTRAINTS.md) — the charter (C1–C8). **Read this first; it binds everything.**
3. [`00-hypothesis-decomposed.md`](docs/00-hypothesis-decomposed.md) — the thesis restated, decomposed into H1–H5.
4. [`01-prior-art-quantum-inspired.md`](docs/01-prior-art-quantum-inspired.md) — the real field, with numbers.
5. [`02-why-classical-not-a-quantum-computer.md`](docs/02-why-classical-not-a-quantum-computer.md) — why not a QC.
6. [`03-complex-amplitudes-and-the-band.md`](docs/03-complex-amplitudes-and-the-band.md) — "not 0/1, a band," answered honestly.
7. [`04-math-and-mechanisms.md`](docs/04-math-and-mechanisms.md) — the implementable equations, each with its cost term.
8. [`05-feasibility-verdict.md`](docs/05-feasibility-verdict.md) — per-claim PROCEED/KILL and the code greenlight gate.
9. [`06-prototype-plan.md`](docs/06-prototype-plan.md) — the fair bake-off design.
10. [`07-pattern-as-token-the-objective.md`](docs/07-pattern-as-token-the-objective.md) — the deepest leg.
11. [`08-combinatorial-capacity.md`](docs/08-combinatorial-capacity.md) — the capacity math, done right.
12. [`09-hardware-requirements.md`](docs/09-hardware-requirements.md) — everything that fits an M1.
13. [`10-frequency-salience-glow.md`](docs/10-frequency-salience-glow.md) — the glow / confidence layer.
14. [`references.md`](docs/references.md) — annotated bibliography.
15. [`REVIEW-LOG.md`](docs/REVIEW-LOG.md) — five adversarial review passes and what each changed.
16. [`implementation/`](implementation/README.md) — the Rust build ladder, value-ranked and kill-gated.
17. [`implementation/METRICS-AND-GATES.md`](implementation/METRICS-AND-GATES.md) — **the testability contract**: every gate as an exact, runnable PASS/FAIL (bits-per-byte, collapse ratios, the invariance margin, ECE, effect sizes, the paired significance test). No "~" thresholds. Read this to check that each claimed number is actually measurable.
18. [`implementation/VERIFICATION.md`](implementation/VERIFICATION.md) + [`implementation/tests/`](implementation/tests/README.md) — **how "green" is made to mean "works"**: the five-layer test pyramid, the anti-fake provenance system (numbers can only be *computed*, never typed), the negative-control battery (tests that must go red on broken code), the mutation catalog, and per-phase task cards a coding agent executes. Read this if your worry is "all tests pass but the model doesn't actually work / doesn't hit the numbers."

---

## What this repo deliberately avoids

- **No pretrained anything** in controlled runs (no BERT/CLIP/SONAR/word2vec/checkpoints) — see C1.
- **No BPE by default** — raw bytes/characters are the default; subword is an explicit *ablation* — see C2.
- **No libtorch/tch/CUDA** for headline numbers — pure-Rust (Burn/Candle) only; GPU is a quarantined perf note — see C3.
- **No imported baseline** — the baseline is *built here*, param-matched, same training loop — see C7.

If you read only two files, read [`GROUND-UP-CONSTRAINTS.md`](docs/GROUND-UP-CONSTRAINTS.md) and
[`05-feasibility-verdict.md`](docs/05-feasibility-verdict.md).

---

### Interview questions this doc answers

- *"Is this a quantum-computing project?"* No. It is classical complex linear algebra; the word
  "quantum" refers to the **math**, run on an M1. A real QC is strictly worse here (see doc 02).
- *"What's actually new versus a normal neural net?"* Two things: **complex phase** as a
  representational axis, and **pattern-as-token** prediction in a shared latent space.
- *"How would you know if it's junk?"* Every claim has a kill number in the verdict table above,
  and the quantum claim dies if a phase=0 ablation ties at equal params.

### Operator's scar

The first version of this README led with "quantum" and buried the classical caveat in a footnote.
Three reviewers independently assumed we were proposing to *buy quantum time*. The lede now states
"classical" in the first sentence, because the single most expensive misunderstanding in this project
is the one where a stakeholder thinks the M1 requirement is a placeholder for a QPU.
