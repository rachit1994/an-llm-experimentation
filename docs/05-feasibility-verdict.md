# 05 — Feasibility verdict (per-claim PROCEED / KILL)

This is the decision doc. Each claim gets a verdict, a **kill criterion** (the measured number that
ends it), and a **revisit trigger** (what would make us re-open a killed claim). It closes with the
**code greenlight gate** and an honest stakeholder framing.

Precedence reminder: this verdict is subordinate to
[`GROUND-UP-CONSTRAINTS.md`](GROUND-UP-CONSTRAINTS.md). A "PROCEED" that violates a constraint is
void.

---

## Per-claim verdicts

### C1 — Quantum formalism (complex amplitudes + interference + unitary + Born)

- **Verdict: PROCEED.**
- **Basis:** Published, competitive at classification scale on modest hardware (Wave Network
  90.91/91.66% single-layer AG News; C-NNQLM complex > real; uRNN solves copy — doc 01). All
  standard complex linear algebra, trainable by plain backprop (doc 04).
- **Kill criterion:** the *phase* degree of freedom must earn its keep — tested via H1 (see the
  "complex phase" row). C1 as a *framework* is not itself killable; its *value* is C1's phase claim.
- **Revisit trigger:** n/a (it's the substrate for C2–C4).

### C2 — Pattern-as-token

- **Verdict: PROCEED, gated at G0.**
- **Basis:** Continuous-latent prediction is a live, promising direction (Coconut, Large Concept
  Models, JEPA — doc 07); ~46% of a small model's params are the vocabulary table it reclaims
  (doc 09).
- **Kill criterion (H2/H3):** representation **collapses** (`erank(Z) < 0.5·erank(Z*)` or
  `meanstd(Z) < 0.5·meanstd(Z*)`, i.e. relative to the *targets* — not the vacuous "10× a constant")
  **or** **bits-per-byte** is **>10% worse** than the param-matched byte baseline. → **STOP the
  project** (this is the load-bearing leg). Exact protocol:
  [`implementation/METRICS-AND-GATES.md`](../implementation/METRICS-AND-GATES.md) §2–§3.
- **Revisit trigger:** a new anti-collapse objective (e.g. a better VICReg/VQ variant) with evidence
  on a public benchmark.

### C3 — Sparse-attractor capacity

- **Verdict: PROCEED.**
- **Basis:** Modern Hopfield = attention has exponential storage capacity (Ramsauer et al. 2020,
  doc 08); a few thousand sparse nodes + `N²/2` weights are M1-sized (doc 09).
- **Kill criterion (H4):** the encoder-invariance margin fails — `within < 0.90` **or**
  `between > 0.30` **or** `within − between < 0.50` at the pre-registered augmentation rates. → narrow
  to "soft similarity retrieval / attractor stability," drop "encoder invariance." Exact protocol §4.
- **Revisit trigger:** higher `β` / better energy landscape / stronger `L_inv` weight recovers the margin.

### C4 — Frequency glow (calibrated confidence)

- **Verdict: PROCEED, narrow-able.**
- **Basis:** Hebbian well-deepening + Bayesian priors are standard; the payoff (inspectable,
  calibrated, on-device confidence) is the product story (doc 10).
- **Kill criterion (H5):** glow does **not** reduce ECE vs a no-glow model (brightness ⟂
  correctness). → narrow to "salience/attention weighting only," drop "confidence/anti-hallucination."
- **Revisit trigger:** a corrected brightness law (better IDF-like salience) restores calibration.

### The decisive one — complex **phase** is the novelty (H1)

- **Verdict: DECISIVE ABLATION, run early (Phase 4).**
- **Kill criterion:** the real-valued (**phase = 0**) ablation **ties** the full-complex model at
  equal params (gap within seed noise). → **drop the "quantum" claim entirely**; keep pattern + glow.
- **Revisit trigger:** phase helps on a *different* task family (e.g. long-range/compositional) even
  if it ties on the first — re-scope rather than re-open blindly.

### Quantum **hardware**

- **Verdict: REJECTED (settled).** Barren plateaus, dequantization, simulability pincer (doc 02).
- **Revisit trigger:** a proven, *non-contrived*, data-rich quantum ML advantage — none exists today.

### Generation **at scale**

- **Verdict: UNPROVEN — gated (Phase 6).** The Born readout `O(|V|·d)` wall (doc 04) is unbroken in
  the quantum-inspired literature (doc 01).
- **Kill criterion:** if no mitigation (low-rank/tied/hierarchical/VQ) gets readout cost and quality
  to parity, → **narrow to encoder-only** (classification/retrieval), which is where the evidence
  already supports us.

---

## Verdict summary

| Claim | Verdict | Kill number | If killed |
|---|---|---|---|
| C1 formalism | **PROCEED** | (phase via H1) | — |
| C2 pattern-as-token | **PROCEED (G0)** | collapse (rel. to targets) OR BPB > 1.10× | STOP project |
| C3 sparse attractor / invariance | **PROCEED** | within<0.90 OR between>0.30 OR margin<0.50 | narrow to attractor-stability |
| C4 glow | **PROCEED (narrow-able)** | ΔECE > −0.02 OR acc drop OR not significant | narrow to salience |
| phase novelty (H1) | **DECISIVE** | Δ < 1.0pt (or <2% BPB) OR not significant | drop phase/"wave" claim |
| quantum hardware | **REJECTED** | — | (stays rejected) |
| generation at scale | **UNPROVEN (gated)** | needs full `O(\|V\|·d)` | narrow to encoder-only |

---

## Can it run on a Mac Mini M1? Yes — with one caveat.

- **Compute:** every "decide" experiment fits the M1 (doc 09): pattern model 12.6M params / ~151MB
  train footprint; AG News ~6–15 min; char-LM ~50–125 min; associative memory is a ~1MB `d×d`
  matrix.
- **The caveat:** Apple **Metal has only partial complex support**. Mitigation (C3/C4): run the
  **CPU (NdArray) backend for correctness first**, and use **`(re, im)` real pairs** everywhere so
  the same code runs on Metal for speed (doc 03, `implementation/00-...`). CPU-first is also what
  makes bit-for-bit determinism (C4) achievable.

---

## Code greenlight gate

Do **not** write Phase-N+1 code until Phase-N's gate passes on a **logged, reproducible** run:

1. **G-stack:** the `(re, im)` complex newtype passes its scalar-oracle + finite-difference gradient
   check (`implementation/00-...`). *Without this, every later number is suspect.*
2. **G0 (Phase 1):** pattern predictor shows **no collapse** (relative to targets) and **BPB ≤ 1.10×**
   the param-matched byte baseline. *This is the project-level go/no-go.*
3. Only then Phases 2–5 (memory, glow, complex, unitary), each behind its own gate.
4. Phase 6 (generation) only if a mitigation beats the `O(|V|·d)` wall.

---

## Honest stakeholder framing

**What to promise:** at **small scale**, a model with (a) a **novel representation** (complex phase
+ pattern-as-token) and (b) **efficiency** (reclaimed vocab params, denser learning signal), plus a
**calibrated, inspectable, on-device confidence** signal that mainstream LLMs don't expose. This is
a *research + on-device-value* story, defensible with the doc-01 numbers and our own ablations.

**What NOT to promise:** beating GPT-class models at open-ended **generation at scale**. That is
unproven, gated behind the Born-readout wall, and contradicted by nobody having done it. Promising
it is how this project loses credibility (and how doc 06's "test against LLMs" gets misread as a
head-to-head).

---

### Interview questions this doc answers

- *"What's your go/no-go, in one gate?"* G0: the next-pattern predictor must not collapse (effective
  rank and per-dim std ≥ 0.5× the targets') and must reach BPB ≤ 1.10× a param-matched byte baseline;
  fail ⇒ stop the project.
- *"Which claim, if it fails, kills the branding vs kills the project?"* Phase=0 tie kills the
  *branding* (drop "quantum"); pattern collapse / >10% perplexity kills the *project*.
- *"Can it really run on an M1?"* Yes; the only caveat is Metal's partial complex support, handled
  by CPU-first + `(re,im)` pairs.

### Operator's scar

An earlier version of this doc had C2 as "PROCEED" with no STOP condition, because nobody wanted to
write the sentence "this could end the project." That omission let the team treat pattern-as-token
as assumed-good and start building Phase 4 in parallel. We now write the STOP condition *first* and
in bold: the load-bearing claim must be the one with the sharpest kill number, or the whole ladder
is built on faith.
