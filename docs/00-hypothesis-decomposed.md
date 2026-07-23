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

Every threshold below is a pre-registered constant with an exact measurement procedure; the full
definitions (BPB, the collapse ratios, the invariance margin, ECE bins, effect sizes, and the paired
significance test over 5 seeds) live in
[`implementation/METRICS-AND-GATES.md`](../implementation/METRICS-AND-GATES.md). No threshold here
contains a "~".

| # | Hypothesis | Metric | **Kill number (exact, pre-registered)** |
|---|---|---|---|
| **H1** | Complex **phase** carries representational information a real-valued net cannot recover at equal params. | `Δ = metric(full-complex) − metric(phase=0)`, param-matched, equal tuning; metric = AG News accuracy / −BPB. | `Δ < 1.0` accuracy pt (or `< 2%` rel. BPB) **or** not significant (paired, `p<0.05`, 5 seeds) ⇒ **drop the phase/"wave" claim** (§6). |
| **H2** | Predicting the **next pattern** is competitive with predicting the next byte at equal `(d,L)`. | **Bits-per-byte** (BPB) of the pattern model (256-way Born head) vs param-matched byte baseline. | `BPB(pattern) > 1.10 · BPB(baseline)` **or** collapse (H3) ⇒ **stop project** (§1,§3). |
| **H3** | Pattern prediction does **not collapse** (constant/low-rank output). | Effective rank and mean per-dim std of predictions **relative to the target patterns**. | `erank(Z) < 0.50·erank(Z*)` **or** `meanstd(Z) < 0.50·meanstd(Z*)` ⇒ **stop** (§2). *(The old "10× a constant" was vacuous: `10·0 = 0`.)* |
| **H4** | **(headline)** the encoder maps *augmented inputs* to the same pattern, and different concepts to different patterns. | `within = sim(enc(aug x), enc x)`, `between = sim(enc x, enc y)` at pre-registered aug rates. | `within < 0.90` **or** `between > 0.30` **or** `within−between < 0.50` ⇒ narrow to attractor-stability (§4). |
| **H5** | **Brightness** (glow) tracks **correctness**, beating a no-glow model on calibration. | ECE (`M=15` bins, held-out) and accuracy, glow vs no-glow. | `ECE(glow) > ECE(no-glow) − 0.02` **or** `acc(glow) < acc(no-glow) − 0.5pt` **or** not significant ⇒ narrow to "salience only" (§5). |

H1 is **decisive** for the phase/"wave" mechanism: if the real-valued ablation does not clear the
effect-size floor with significance, the project keeps the (real-valued) VSA encoder and glow but drops
the wave/phase claim. H2/H3 gate the pattern objective (the load-bearing leg). H4 is the **headline
invariance** claim, gated with a discrimination margin so it cannot be won by collapse, and *trained*
by an explicit `L_inv` term (§4.4) so that following the plan actually produces it. H5 gates the
product story.

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
