# Phase 2 — Sparse attractor memory (the "bee test")

**Goal:** show that a concept, stored as a **sparse distributed pattern**, **reliably re-triggers
the same attractor** under input noise — "bee always fires (nearly) the same nodes." This is C3 /
H4, and it is what makes the capacity story (doc 08) and the glow story (doc 10) real rather than
theoretical.

Depends on: Phase 1 passing G0 (there is no point storing patterns the objective can't learn).
Blocks: Phase 3 (glow needs memory) and Phase 6 (generation retrieves concepts).

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

## The bee test (the metric that decides Phase 2)

For each stored concept `c` with clean pattern `ξ_c`:

1. Corrupt it: `ξ̃_c = noise(ξ_c, level)` (flip/drop a fraction of active nodes, add Gaussian jitter).
2. Retrieve: `ξ̂_c = retrieve(ξ̃_c)`.
3. **Hit** if the active-node overlap between `ξ̂_c` and `ξ_c` exceeds a threshold (e.g. Jaccard /
   node-overlap ≥ target).

```
bee-test hit-rate = fraction of concepts that retrieve back to themselves under noise
PASS  ⟺  hit-rate ≥ ~90%  at the target noise level
```

Second, softer check — **overlap ∝ similarity**: *similar* concepts should retrieve to *nearby*
attractors (graceful, meaningful confusion), not random collapse. Plot retrieved-overlap vs
input-similarity; it should be monotone. A model that passes the hit-rate but has *random* confusions
has a broken geometry.

---

## The gate

```
PASS      ⟺   bee-test hit-rate ≥ ~90%  AND  overlap-vs-similarity is monotone
NARROW    ⟹   hit-rate < ~90% (unstable attractors)
              → drop "reliable attractor storage", keep "soft similarity retrieval"
                (a weaker but still-honest claim: retrieval returns *near* the concept, not *the* concept)
```

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
  pattern, retrieve, and require ≥ ~90% node-overlap back to the original — plus overlap-vs-similarity
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
