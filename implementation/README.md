# implementation/ — the Rust build ladder (value-ranked, kill-gated)

This is the *build*, ordered so that the **highest-value, cheapest-to-kill** work happens first.
The organizing principle: **buy information as early and as cheaply as possible**. If the project is
going to die, it should die in week 3 on a $0 M1 run, not in month 4 after a generation stack.

Precedence: [`../GROUND-UP-CONSTRAINTS.md`](../GROUND-UP-CONSTRAINTS.md) outranks every phase doc
here. A phase that violates a constraint is wrong, not the constraint.

---

## Phase table (highest-value-and-cheapest-to-kill first)

| Phase | What it proves | Kill/narrow gate | Why this order |
|---|---|---|---|
| **[Phase 1](PHASE-1-pattern-feasibility.md)** — pattern feasibility | The load-bearing leg (C2) trains without collapsing, competitive perplexity | **G0**: collapse OR >10% perplexity ⇒ **STOP PROJECT** | Most valuable, most killable, cheapest to run (~hours). Do it first. |
| **[Phase 2](PHASE-2-sparse-attractor-memory.md)** — sparse attractor memory | Concepts re-trigger reliably (C3, the "bee test") | Hit-rate < ~90% ⇒ narrow to soft retrieval | Needed by glow (Phase 3) and generation (Phase 6). |
| **[Phase 3](PHASE-3-glow-confidence.md)** — glow / confidence | Calibrated inspectable confidence (C4) | ECE not beaten ⇒ narrow to salience | The product story; cheap once memory exists. |
| **[Phase 4](PHASE-4-complex-wave.md)** — complex wave | Phase earns its keep (C1/H1) | phase=0 ties ⇒ **drop "quantum"** | The decisive branding ablation; run once the harness is trusted. |
| **[Phase 5](PHASE-5-unitary-dynamics.md)** — unitary dynamics | Long-range memory (uRNN leg) | no gain ⇒ **drop unitary** | Builds on complex (Phase 4). |
| **[Phase 6](PHASE-6-generation-multimodal.md)** — generation / multimodal | Beat the `O(\|V\|·d)` wall | needs full `O(\|V\|·d)` ⇒ narrow to encoder-only | Most expensive, most uncertain; last. |

---

## Dependency graph

```
        ┌──────────────► Phase 2 (memory) ──┐
        │                                     ├──► Phase 6 (generation/multimodal)
 Phase 0 (stack) ──► Phase 1 (pattern) ──────┤
        │                                     │
        │                └► Phase 3 (glow) ◄──┘  (glow needs memory)
        │
        └──► Phase 4 (complex) ──► Phase 5 (unitary)

 Edges: 0 → 1 → {2,3,4};  4 → 5;  {1,2} → 6.
```

Read as: everything depends on the **stack** (Phase 0). Phase 1 (pattern) is the gate to Phases 2,
3, 4. Phase 3 (glow) also needs Phase 2 (memory). Phase 5 needs Phase 4. Phase 6 needs both the
pattern objective (1) and the memory (2).

---

## Effort roll-up (calendar estimates, single engineer on an M1)

| Milestone | Scope | Estimate |
|---|---|---|
| **Kill-or-prove the core** | Phase 0 (stack + gradient-checked kernels) + Phase 1 (G0) | **~3–4 weeks** |
| **Full core** | + Phases 2, 3, 4 (memory, glow, complex/H1) | **~6–7 weeks** |
| **Generation** | + Phases 5, 6 (unitary, Born generation past the wall) | **~11–15 weeks** |

The first milestone is the whole bet: in **3–4 weeks** you know whether pattern-as-token trains
without collapsing at competitive perplexity. If G0 fails, you stop having spent ~one month, not one
quarter.

---

## STOP conditions (project-kill vs claim-narrow)

Be explicit about which failures **kill the project** and which merely **narrow a claim**:

**Project-kill (stop everything):**
- **G0 fails** — the pattern objective collapses or is >10% worse perplexity than the param-matched
  token baseline (Phase 1). The load-bearing leg is gone; there is no project without it.

**Claim-narrow (keep going, drop one claim):**
- **Phase 2** bee-test < ~90% ⇒ drop "reliable attractor," keep "soft similarity retrieval."
- **Phase 3** glow doesn't beat no-glow on ECE ⇒ drop "calibrated confidence," keep "salience."
- **Phase 4** phase=0 ties at equal params ⇒ **drop the "quantum" claim**, keep pattern + glow.
- **Phase 5** unitary shows no long-range gain ⇒ drop the unitary layer.
- **Phase 6** generation needs full `O(|V|·d)` with no working mitigation ⇒ **narrow to
  encoder-only** (classification/retrieval), which the evidence already supports.

The asymmetry is deliberate: only the pattern objective can kill the project; everything else
degrades gracefully into a smaller, still-honest claim.

---

## The discipline that makes the ladder trustworthy

- **Gate before you climb** (doc 05 code greenlight): no Phase N+1 code until Phase N's gate passes
  on a logged, bit-for-bit-reproducible run.
- **One switch per comparison** (C6): every number is attributable to one toggled mechanism.
- **Gradient-check every kernel** (Phase 0): a scalar oracle + finite-difference check per complex
  op, or the later numbers are built on sand.
- **CPU-first source of truth** (C3/C4): headline numbers on NdArray/CPU; Metal is a separately
  verified speed note.

---

### Interview questions this doc answers

- *"What do you build first and why?"* Phase 1 (pattern feasibility): most valuable, most killable,
  runs in hours; its G0 gate is the project go/no-go, so you buy the decisive information cheapest.
- *"When would you stop the whole project?"* Only at G0 — collapse or >10% perplexity gap. Every
  other failure narrows a claim rather than killing the project.
- *"How long until you know if this works?"* ~3–4 weeks to kill-or-prove the core (Phase 0 + G0).

### Operator's scar

The first plan built the complex/unitary machinery (Phases 4–5) *first* because it was the most fun,
then discovered in month 3 that the pattern objective collapsed — making all the elegant complex
kernels moot. We reordered the entire ladder around one question: *what is the cheapest experiment
that can kill the project?* That is Phase 1, and it now runs before anything beautiful gets built.
