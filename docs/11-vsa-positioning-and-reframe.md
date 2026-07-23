# 11 — Positioning: this is a learned Vector Symbolic Architecture (and why the reframe)

This doc records a **framing correction** made after the first draft, and the prior-art field the
project actually belongs to. It changes **no math and no numbers** — docs 03, 04, 07, 08, 10 all
stand — but it changes what we *lead with*, because the first framing led with the mechanism and
read as style over substance.

---

## The correction

**v1 framing:** "Amplitude Language Model" — the title was the *mechanism* (complex/wave math). A
reviewer correctly reads a mechanism-as-identity title as authors more attached to a technique than
a problem. It also (a) narrowed the work to "language" when the representational claim is
modality-general, and (b) invited a fight about the word "quantum" that the project does not need.

**v2 framing:** lead with the **problem** — a *learned, from-scratch encoder that maps inputs to
stable, surface-invariant, inspectable concept patterns* — and name the field it belongs to:
**Vector Symbolic Architectures (VSA) / Hyperdimensional Computing (HDC)**. The wave/amplitude math
is then a **mechanism** (frequency-domain binding), demoted from identity and contingent on one
ablation (H1).

Precedence unchanged: [`GROUND-UP-CONSTRAINTS.md`](GROUND-UP-CONSTRAINTS.md) still binds everything.

---

## The intuition was right — it has a rigorous, 35-year-old home

The originating intuition — *"a fixed but expandable substrate whose **nodes and weights**
combine/**permute** to produce nearly the same **pattern** for a given input"* — is, almost verbatim,
**Vector Symbolic Architectures** (Kanerva; Plate; Gayler; Smolensky; survey Kleyko et al. 2023).
VSA represents concepts as high-dimensional, **near-orthogonal** vectors and builds structure from
exactly three algebraic operations:

| VSA operation | What it does | In this project |
|---|---|---|
| **Bind** `a ⊛ b` | associate a role with a filler (circular convolution / Hadamard / XOR) | wave-domain modulation `ψ₁⊙ψ₂` (doc 04); *the* mechanism |
| **Superpose** `a + b` | form a set/multiset (vector addition) | interference / superposition (doc 04) |
| **Permute** `Πa` | encode order/position (fixed invertible reindexing) | a fixed permutation (a special orthogonal map, doc 04's unitary family) |

The single most important sentence: **"permutation" is a named, canonical VSA primitive, not a
metaphor.** The intuition did not wander off into cool-sounding territory; it re-derived the
operations of an established field. That is the biggest credibility upgrade available to the project —
it converts "novice reaching for a vibe" into "positioned in a literature with capacity theorems."

---

## The identity that saves the wave math (and demotes it honestly)

Why keep complex amplitudes at all, once we admit the substrate is VSA? Because the wave math *is*
one particular VSA binding, viewed in the frequency domain. Plate's **Holographic Reduced
Representations** bind by **circular convolution** `a ⊛ b`. By the circular convolution theorem:

```
F(a ⊛ b) = F(a) ⊙ F(b)          (elementwise product of the spectra)
```

Each spectral coordinate `F(a)_k = ρ_k e^{iθ_k}` multiplies magnitudes and **adds phases** — which is
exactly the modulation `ψ₁⊙ψ₂ = r₁r₂ e^{i(φ₁+φ₂)}` of doc 04. So:

- Working **directly in the amplitude (wave) domain, our binding = VSA binding.**
- A **phase-only** (unit-magnitude) state is Plate's *circular* HRR; the Born-normalized state is its
  unit-norm case.
- **Unbinding** (querying a bound pair) is circular correlation = multiply by the complex
  **conjugate** = phase **subtraction**.

Consequence: the complex mathematics is a *mechanism for a distributed-binding representation*, not a
separate thesis. Dropping it (the `phase=0` ablation, H1) reduces the model to a real-valued
distributed code and asks the honest question: **did phase-domain binding earn its parameters?**

---

## What is actually novel (the narrow, defensible claim)

Leading with VSA means entering a **mature, crowded field**, so novelty must be stated narrowly. It is
the *intersection* that is unoccupied:

> A **learned, from-scratch** (C1) VSA-style encoder whose binding is realized in the **wave/frequency
> domain**, whose patterns are **learned attractors** (modern Hopfield, doc 08), carrying a
> **calibrated, inspectable confidence** signal (glow, doc 10) — every mechanism individually
> ablatable (C6).

- Classical VSA codebooks and binding maps are **hand-designed and un-calibrated** → we learn them.
- CLIP / ImageBind / LCM shared-embedding alignment relies on **pretrained** encoders → we forbid that
  (C1).
- Neither classical VSA nor shared-embedding models expose a **calibrated confidence** → the glow layer
  does (doc 10).

Read any one axis alone and the work looks derivative; the claim lives only in the conjunction.

---

## The one hazard the reframe must not smuggle back in

"Nearly the same pattern for an image, a sentence, or a video" is the **ImageBind / shared-embedding**
problem. It is real and worth solving, but it **collides with C1 and the M1 budget**: from-scratch
cross-modal alignment cannot be demonstrated cheaply and honestly. Therefore:

- **Testable claim (H4):** *single-modality* invariance — same concept under paraphrase / augmentation
  / byte-noise → nearly the same pattern. Decidable on an M1 from scratch.
- **Destination (not a result):** cross-modal invariance. Stated as future scaling, never as a demoed
  claim (doc 09's ladder, whitepaper §10).

Making multimodality the *headline* would relocate the exact "look cool, can't deliver" trap the
reframe exists to remove.

---

## Naming

The rename is deliberate and descriptive, not another brandable acronym (acronym-first naming is the
"look cool" tell). Working title: **Convergent Concept Representations** — the property (many surface
forms *converge* to nearly the same concept pattern) rather than the mechanism. It is a placeholder;
any accurate, boring name is acceptable. What matters is that the title names *what the thing does*,
not *how the arithmetic works*.

---

### Interview questions this doc answers

- *"Isn't 'amplitude language model' just branding?"* Yes — that was the v1 error. The work is a
  **learned Vector Symbolic Architecture**; the wave math is the frequency-domain form of VSA binding,
  and it lives or dies on one ablation (H1).
- *"Where does this sit in the literature?"* VSA/HDC (Kanerva, Plate, Smolensky) for the substrate;
  modern Hopfield for storage; JEPA/Coconut/LCM for the predictive objective; CLIP/ImageBind for the
  (quarantined, pretrained) shared-space precedent.
- *"What's genuinely new, given all that prior art?"* The intersection: learned + from-scratch +
  wave-domain binding + learned attractors + calibrated confidence, each individually ablatable.

### Operator's scar

The v1 title put "Amplitude" first and a reader we respect asked, in effect, "is this someone who
wants to look clever, or someone solving a problem?" — before reading a single equation. The title had
already lost the argument. The scar: *name the problem, not the trick.* The trick (wave-domain binding)
is real and stays in the paper — as §3.3, where it belongs, contingent on H1 — not on the cover.
