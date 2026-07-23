# Phase 2 — Sparse attractor memory (the "bee test")

**Goal:** show that a concept, stored as a **sparse distributed pattern**, **reliably re-triggers
the same attractor** under input noise — "bee always fires (nearly) the same nodes." This is C3 /
H4, and it is what makes the capacity story (doc 08) and the glow story (doc 10) real rather than
theoretical.

Depends on: Phase 1 passing G0 (there is no point storing patterns the objective can't learn).
Blocks: Phase 3 (glow needs memory) and Phase 6 (generation retrieves concepts).

**Test suite (execute this):** [`tests/PHASE-2.md`](tests/PHASE-2.md) — the invariance metric plus the
two controls that make it non-gameable (`nc_collapsed_invariance`, `nc_invariance_needs_Linv`) and the
`kae_invariance` planted-concept fixture.

---

## What we build

1. **Sparse k-of-N coding.** Concepts are represented as patterns with only `k` of `N` nodes active
   (`k/N ≈ 1.5–2%`, doc 08). A `k`-winners-take-all (top-`k`) activation enforces the sparsity;
   `N` a few thousand, `k` a few tens.
2. **Modern-Hopfield retrieval** (Ramsauer et al., doc 08) — the attractor dynamics *are* attention:
   ```
   retrieve(ξ) = X · softmax(β · Xᵀ ξ) ,     β = 1/√d
   ```
   `X` = stored patterns (columns), `ξ` = noisy query, `β = 1/√d` = inverse-temperature. One step
   pulls a noisy query toward the nearest stored attractor; iterating sharpens it. This is a
   `qilm-core` kernel with a scalar oracle + gradient check (Phase 0 discipline).

---

## Two distinct tests (do not conflate them)

All exact thresholds are pre-registered in [`METRICS-AND-GATES.md`](METRICS-AND-GATES.md) §4.

- **(4a) Attractor stability** — a *code-space* denoising test. Corrupt a stored code and require it to
  retrieve itself. This is the memory *mechanism*.
- **(4b) Encoder invariance** — the *headline* (H4): the **encoder** maps an *augmented input* to the
  same pattern (`enc(aug(x)) ≈ enc(x)`). This is the property the model is named for. It is a
  property of the encoder + the `L_inv` training term (protocol §4.4), not of retrieval alone.

### (4a) Attractor stability

For each stored concept `c` with clean pattern `ξ_c`: corrupt `ξ̃_c = noise(ξ_c, σ=0.10·meanstd(ξ))`,
retrieve `ξ̂_c = retrieve(ξ̃_c)`, and score a **hit** if `sim(ξ̂_c, ξ_c) ≥ 0.90` (Jaccard for sparse,
cosine for dense).

```
stability PASS  ⟺  hit-rate ≥ 0.90 at code-noise σ = 0.10·meanstd(ξ)   (over 5 seeds, mean)
```

### (4b) Encoder invariance + discrimination (the H4 gate)

Invariance alone is gameable by collapse, so the gate is a **margin** between same-item and
different-item similarity (protocol §4.3), evaluated at the pre-registered augmentation rates
{`delete 10%`, `substitute 10%`, `case 50%`, `swap w=3`}:

```
within  = mean_x  sim(enc(aug(x)), enc(x))          # same item, augmented
between = mean_{x≠y} sim(enc(x), enc(y))            # different items

PASS (H4)  ⟺  within ≥ 0.90  AND  between ≤ 0.30  AND  margin = within − between ≥ 0.50
NARROW     ⟹  any condition fails → drop "encoder invariance", keep "attractor stability" (4a only)
```

A collapsed encoder (`within ≈ between ≈ 1`, `margin ≈ 0`) **fails** by construction — that is the
point of the margin. The `no_invariance` ablation (train without `L_inv`) must also fail this gate, which
proves the invariance is *caused by the objective* and not incidental.

Before narrowing, try the obvious recovery: **raise `β`** (sharper attractors) and check the energy
landscape. Modern Hopfield's capacity is exponential in `d` (doc 08), so instability at a few
thousand nodes usually means `β` is too low or `k` too large, not that the idea is wrong.

---

## What "stable" means here (and why it's not free)

- **Stable:** small input perturbations map to the *same* fixed point (deep, well-separated basins).
- **Unstable:** perturbations cross basin boundaries → the concept retrieves to a *neighbor* or a
  spurious mixture. Symptoms: hit-rate falls off a cliff as noise rises; overlap-vs-similarity is
  non-monotone.

Stability costs **separation** between stored patterns (sparsity helps: sparse patterns are more
near-orthogonal, doc 08's JL argument) and an adequately high `β`. It is the classic capacity ↔
stability tradeoff, and the bee test measures exactly where you sit on it.

---

## Instrumentation

- Hit-rate vs noise-level curve (the headline plot).
- Overlap-vs-similarity scatter (geometry sanity).
- Basin depth / energy at stored patterns vs random points.
- Spurious-attractor count (how many retrievals land on *nothing* stored).

---

### Interview questions this doc answers

- *"How do you test that a concept is stored, not just named?"* The bee test: corrupt a concept's
  pattern, retrieve, and require ≥ 0.90 node-overlap back to the original — plus overlap-vs-similarity
  monotonicity so confusions are graceful.
- *"What retrieval rule?"* Modern-Hopfield / attention: `X·softmax(β XᵀΞ)`, `β = 1/√d` — one step is
  one attention step (doc 08).
- *"What if attractors are unstable?"* First raise `β` / increase sparsity; if it still fails, narrow
  from "reliable attractor" to "soft similarity retrieval."

### Operator's scar

Our first bee test reported 97% and we celebrated — until the overlap-vs-similarity plot showed the
retrievals were *random*: every noisy query fell into the **same** giant spurious attractor that
happened to overlap most concepts. High hit-rate, dead geometry. The scar: *never accept the
hit-rate without the geometry plot*, because a single dominant basin can fake reliability while
storing nothing distinguishable.
