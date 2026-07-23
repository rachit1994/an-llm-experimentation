# an-llm-experimentation

A **ground-up** language-model experiment. Not a fork of the transformer stack, not a
wrapper around a pretrained checkpoint, and not a quantum-hardware project. It is a
disciplined test of a *different base for language modeling*, implemented in **classical**
complex linear algebra and sized to run on a **Mac Mini M1**.

The repo is a dossier first and a build ladder second. Every claim below is decomposed into
an individually falsifiable hypothesis with a *killable number*, and every mechanism is
switchable so that any reported result is attributable to one cause.

---

## One-paragraph verdict

Representing each linguistic unit as a **complex probability amplitude** (magnitude + phase),
composing units by **wave interference**, evolving them with **unitary** maps, and reading them
out with the **Born rule** is a legitimate, *already-published* direction (Wave Network reaches
**90.91%**/**91.66%** single-layer on AG News vs a single transformer layer's ~**71–72%**,
approaching BERT-base's **94.64%**). The genuinely novel legs are (1) **complex phase** as a
representational degree of freedom that real-valued nets lack, and (2) **pattern-as-token** —
predicting the *next distributed activation pattern* in a shared latent space rather than the
next discrete vocabulary symbol. Both are **PROCEED** at small scale on an M1; **generation at
scale is unproven** and gated behind the Born-readout `O(|V|·d)` wall. A real quantum computer is
the *wrong* tool here (barren plateaus, dequantization, the simulability pincer), which is exactly
why the whole thing runs classically. See [`05-feasibility-verdict.md`](docs/05-feasibility-verdict.md).

---

## Theory → quantum formalism → classical implementation

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

0. [`WHITEPAPER.md`](WHITEPAPER.md) — the peer-review-grade research paper (theory, derivations, capacity bounds, all calculations, references). Start here if you want the rigorous end-to-end argument; the docs below are the working dossier behind it.
1. [`GROUND-UP-CONSTRAINTS.md`](docs/GROUND-UP-CONSTRAINTS.md) — the charter (C1–C8). **Read this first; it binds everything.**
2. [`00-hypothesis-decomposed.md`](docs/00-hypothesis-decomposed.md) — the thesis restated, decomposed into H1–H5.
3. [`01-prior-art-quantum-inspired.md`](docs/01-prior-art-quantum-inspired.md) — the real field, with numbers.
4. [`02-why-classical-not-a-quantum-computer.md`](docs/02-why-classical-not-a-quantum-computer.md) — why not a QC.
5. [`03-complex-amplitudes-and-the-band.md`](docs/03-complex-amplitudes-and-the-band.md) — "not 0/1, a band," answered honestly.
6. [`04-math-and-mechanisms.md`](docs/04-math-and-mechanisms.md) — the implementable equations, each with its cost term.
7. [`05-feasibility-verdict.md`](docs/05-feasibility-verdict.md) — per-claim PROCEED/KILL and the code greenlight gate.
8. [`06-prototype-plan.md`](docs/06-prototype-plan.md) — the fair bake-off design.
9. [`07-pattern-as-token-the-objective.md`](docs/07-pattern-as-token-the-objective.md) — the deepest leg.
10. [`08-combinatorial-capacity.md`](docs/08-combinatorial-capacity.md) — the capacity math, done right.
11. [`09-hardware-requirements.md`](docs/09-hardware-requirements.md) — everything that fits an M1.
12. [`10-frequency-salience-glow.md`](docs/10-frequency-salience-glow.md) — the glow / confidence layer.
13. [`references.md`](docs/references.md) — annotated bibliography.
14. [`REVIEW-LOG.md`](docs/REVIEW-LOG.md) — five adversarial review passes and what each changed.
15. [`implementation/`](implementation/README.md) — the Rust build ladder, value-ranked and kill-gated.

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
