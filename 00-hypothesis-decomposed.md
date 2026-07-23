# 00 — The hypothesis, decomposed

## The theory, restated faithfully

Language models today are continuous-float networks that predict the **next discrete token** from
a fixed vocabulary. This experiment proposes a different base:

> Represent each linguistic unit as a **complex probability amplitude** — a magnitude *and* a
> phase. Let units **interfere** like waves (reinforcing or cancelling), **evolve** them with
> information-preserving **unitary** maps, and **read them out** with the **Born rule**
> (probability = squared magnitude). Make the unit of language the **distributed activation
> pattern** a concept evokes — "a bee" is the characteristic ensemble that fires — and predict the
> **next pattern**, decoding to an actual word only at the moment of emission. Store concepts as
> **sparse attractors** so the same concept reliably re-triggers nearly the same nodes, and let a
> pattern **brighten** (deeper attractor, higher prior) the more often it is encountered, yielding
> a **calibrated, inspectable confidence** signal.

Everything is implemented **classically** in standard complex linear algebra and sized for a
**Mac Mini M1**. "Quantum" names the *mathematics*, not the hardware.

---

## Two honest corrections baked into the thesis

1. **"Not a band vs 0/1."** Neural nets are *already* continuous floats — there is no "0/1 vs a
   band" novelty to be had on magnitude alone. The genuinely new degree of freedom is complex
   **phase**. See [`03-complex-amplitudes-and-the-band.md`](03-complex-amplitudes-and-the-band.md).
2. **"Quantum means the math on an M1."** A real quantum computer is *strictly worse* for this
   (barren plateaus, dequantization, the simulability pincer), which is *why* we run classically.
   See [`02-why-classical-not-a-quantum-computer.md`](02-why-classical-not-a-quantum-computer.md).

---

## The 3-layer model (keep these separate or you will fool yourself)

| Layer | Question it answers | Failure if you confuse it with another layer |
|---|---|---|
| **Model** (the idea) | *What* is language, representationally? Patterns, phases, attractors. | Claiming a hardware win for an idea-level property. |
| **Numbers** (the math) | *How* do we compute it? Complex linear algebra, Born rule, unitary maps. | Assuming the math needs quantum hardware to be "real." |
| **Substrate** (the machine) | *Where* does it run? A classical M1 CPU/GPU. | Treating the M1 as a stand-in for a future QPU. |

The whole project is: a **Model** expressed in quantum **Numbers** executed on a classical
**Substrate**. The three are independent choices, and the interesting claims live at the Model and
Numbers layers, not the Substrate.

---

## Decomposition into claims (C1–C4)

- **C1 — Quantum formalism, not hardware.** Units are complex amplitudes `α = r·e^{iφ}`; they
  compose by interference; they evolve by unitary maps; they read out via the Born rule. All
  standard complex linear algebra on a classical machine.
- **C2 — Pattern-as-token.** The unit is the distributed *pattern*; the objective predicts the
  **next pattern**, not the next symbol; decoding to a word happens only at emission ("think in
  superposition, measure to speak").
- **C3 — Combinatorial sparse-attractor capacity.** Concepts are sparse distributed patterns
  stored as attractors; the same concept reliably re-triggers nearly the same nodes.
- **C4 — Frequency glow.** A pattern brightens (deeper well, higher prior) with exposure —
  Hebbian well-deepening + Bayesian evidence — giving a calibrated, inspectable confidence signal.

These map one-to-one onto the phase docs 04/07 (C1/C2), 08 (C3), and 10 (C4).

---

## Falsifiable hypotheses (each with a killable number)

Every hypothesis is stated so that a single measured number can *end* it. No hypothesis is allowed
to be "directionally encouraging."

| # | Hypothesis | Metric | **Kill number** |
|---|---|---|---|
| **H1** | Complex **phase** carries representational information a real-valued net cannot recover at equal params. | Accuracy/perplexity gap between full-complex model and its **phase=0** ablation, param-matched. | Gap ≤ **1σ of the seed noise** (i.e. real ablation *ties*) ⇒ drop the quantum claim. |
| **H2** | Predicting the **next pattern** is competitive with predicting the next token at equal params. | Validation perplexity (via a fixed decoder) of pattern model vs param-matched token baseline. | Pattern model **>10%** worse perplexity, **or** representation collapses. |
| **H3** | Pattern prediction does **not collapse** (does not learn a constant/low-rank output). | **Effective rank** / per-dim variance of predicted patterns over the val set. | Effective rank **< 50%** of embedding dim, or variance within **10×** of a constant predictor. |
| **H4** | A concept **reliably re-triggers** the same sparse attractor under input noise ("bee test"). | Node-overlap hit-rate between clean and noisy triggerings of the same concept. | Hit-rate **< ~90%** at the target noise level. |
| **H5** | **Brightness** (glow) tracks **correctness**, giving calibration better than a no-glow model. | Expected Calibration Error (ECE) and reliability-diagram slope, glow vs no-glow. | Glow does **not** reduce ECE vs no-glow (brightness ⟂ correctness) ⇒ narrow the glow claim to "salience only." |

H1 is the **decisive** one for the "quantum" branding: if the real-valued ablation ties, the
project keeps pattern-as-token and glow but drops the quantum-formalism claim. H2/H3 gate the
pattern objective (the deepest and most valuable leg). H4 gates memory. H5 gates the product story.

---

## Why quantum *math* might actually fit language

This is a hypothesis, not a foregone conclusion. Two independent reasons it is worth testing:

1. **Phase and interference as representation.** Language is full of *context-dependent
   reinforcement and cancellation*: a word's contribution to meaning flips with context (negation,
   irony, scope). Complex phase gives a native handle on "same magnitude, opposite effect":
   `|ψ₁+ψ₂|² = r₁²+r₂²+2r₁r₂cos(φ₁−φ₂)` — two units of equal strength can add (in phase) or cancel
   (out of phase) depending only on their **relative phase**. Real-valued nets must *learn* this
   with extra parameters; complex nets get it in the algebra. This is the Wave Network intuition
   (doc 01): magnitude ≈ global semantics, phase ≈ token↔context relationship.
2. **Quantum cognition.** Human judgments show **order effects** (P(A then B) ≠ P(B then A)) and the
   **conjunction fallacy** (P(A∧B) judged > P(A)) that violate classical probability but are
   modeled cleanly by *non-commuting projective measurements* in a Hilbert space (Busemeyer &
   Bruza, *Quantum Models of Cognition and Decision*, 2012; see [`references.md`](references.md)).
   If human meaning-composition is *natively* quantum-probabilistic, a quantum-formalism model may
   need fewer parameters to express the same phenomena. This does **not** claim the brain is a
   quantum computer — only that the *algebra* of order-dependence is a good fit.

Neither reason is proof. Both are the reason H1 exists: we test whether the phase degree of freedom
*earns its keep* at equal parameters, and we kill the branding if it does not.

---

### Interview questions this doc answers

- *"State your hypothesis so I could disprove it."* Five hypotheses H1–H5, each with a single
  measured kill number; H1 (the phase=0 ablation) is the decisive one for the quantum claim.
- *"Why might complex math help with language specifically?"* Phase gives native reinforcement/
  cancellation, and human judgment shows order effects the Hilbert-space formalism models directly.
- *"What are the three layers people conflate?"* Model (the idea), Numbers (the math), Substrate
  (the machine); the interesting claims are at Model/Numbers, and the M1 is not a QPU placeholder.

### Operator's scar

The original hypothesis list had a sixth item — "the model will be more interpretable" — with no
number attached. It survived two planning meetings purely on vibes and quietly justified scope
creep. We deleted it and made the rule: *a hypothesis without a kill number is a slogan.* H5 is
what "interpretable" became once we forced it to name a metric (ECE) and a way to lose.
