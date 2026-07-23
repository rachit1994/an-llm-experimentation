# Phase 6 — Generation & multimodal (beating the O(|V|·d) wall)

**Goal:** the hardest, most uncertain leg — make **Born-rule generation** affordable past the
`O(|V|·d)` readout wall (doc 04), feed continuous patterns back in (Coconut-style), and sketch the
multimodal shared space with **from-scratch** encoders. Run last; narrow to encoder-only if the wall
holds.

Depends on: Phase 1 (the pattern objective) and Phase 2 (attractor memory to retrieve concepts).
Blocks: nothing — this is the top of the ladder.

**Test suite (execute this):** [`tests/PHASE-6.md`](tests/PHASE-6.md) — measured readout costs
(`kat_readout_cost`), per-mitigation (BPB, cost) frontier points, and the encoder-only fallback as a
*measured* verdict.

---

## The wall, restated

Born generation readout is `p_k = |⟨e_k|Oψ⟩|²` for **every** `k ∈ V`, i.e. **`O(|V|·d)` per step**
and an output matrix of `|V|·d` params (doc 04). For `|V|=32000, d=512` that is ~16.4M multiply-adds
*and* ~16.4M params **per step** — the same wall a large-vocab softmax hits, and unbroken in the
quantum-inspired literature (doc 01). This is the leg most likely to force a **narrow to
encoder-only**.

---

## Mitigations (each an ablation, C6; measure quality-vs-cost)

| Mitigation | Mechanism | Cost after | Risk |
|---|---|---|---|
| **Low-rank** output `O = UVᵀ` | factor `\|V\|×d` through rank `k` | `O(k(\|V\|+d))` | rank too low → quality loss |
| **Tied** embeddings | reuse input table as output (doc 07's reclaimed params) | halves vocab params | ties input/output geometry |
| **Hierarchical** readout | tree over `V` | `O(d·log\|V\|)` per step | tree imbalance → skew |
| **VQ codebook** | predict a code index; decode later | readout over `\|codebook\| ≪ \|V\|` | quantization error |

The evaluation is a **quality-vs-cost curve** (Pareto, doc 06): for each mitigation, plot generation
quality (perplexity / sample quality) against readout cost (FLOPs, params, latency). "Beating the
wall" means finding a mitigation on the acceptable-quality frontier at sub-`O(|V|·d)` cost.

---

## Coconut-style feedback (think in patterns, measure to speak)

Rather than decoding to a token at every step and re-embedding it, feed the **continuous predicted
pattern** `ẑ` back in as the next input (doc 07; Coconut, arXiv:2412.06769). Generation then runs in
**pattern space**, carrying multiple continuations in superposition, and only **collapses to a word
at emission** (Born readout at the end, not every step). This *amortizes* the `O(|V|·d)` wall — you
pay it at emission points, not every internal step — which is a second, orthogonal way to soften the
wall alongside the readout mitigations above.

---

## Multimodal shared space (from scratch — C1)

The "one screenshot → image / voice / text" vision (doc 08) needs a **shared amodal concept space**
with per-modality **encoders in** and **decoders out**. Under **C1**, in controlled runs these
encoders/decoders are **trained from scratch**; pretrained CLIP/ImageBind/SONAR are **quarantined**
(motivation and upper-bound reference only, never a controlled number — GROUND-UP-CONSTRAINTS
quarantine protocol).

Scope honestly: full from-scratch multimodal training is beyond an M1's *decision* budget for
anything but a **toy** (small images, short audio, short text) that demonstrates the *shared-space
mechanism*, not competitive multimodal quality. The Phase-6 deliverable is the *mechanism working on
a toy*, with the scaling of quality left to the ladder in doc 09 — never a claim of production-grade
multimodal generation on a Mac Mini.

---

## The gate

```
PROCEED (generation claim)     ⟺  a mitigation reaches acceptable quality at sub-O(|V|·d) cost
                                   (on the quality-vs-cost frontier)
NARROW to encoder-only         ⟸  generation needs the full O(|V|·d) with no working mitigation
                                   → ship the classification/retrieval (encoder) story, which the
                                     evidence already supports (docs 01, 05), and shelve generation
```

Narrowing to encoder-only is an **honest, defensible outcome**, not a failure: it lands the project
exactly where the prior art is strongest (classification/matching, doc 01) and where the on-device
confidence story (doc 10) already delivers value.

---

### Interview questions this doc answers

- *"What's the hardest unsolved problem here?"* Generation past the `O(|V|·d)` Born-readout wall —
  unbroken in the quantum-inspired literature. We attack it with low-rank/tied/hierarchical/VQ
  readout plus Coconut-style pattern feedback that amortizes the cost to emission points.
- *"What if generation doesn't work?"* Narrow to **encoder-only** (classification/retrieval) — the
  regime the evidence already supports and where the confidence story delivers value. An honest land,
  not a failure.
- *"How is multimodality kept honest?"* Shared amodal space with **from-scratch** encoders/decoders
  (C1); pretrained CLIP/ImageBind/SONAR are quarantined; the M1 target is a mechanism-demonstrating
  toy, not production quality.

### Operator's scar

We prototyped a full 32k-vocab Born readout "just to see it work" and it dominated both the FLOP
budget and the parameter count — the model was mostly an output matrix, which is the exact thing
pattern-as-token (doc 07) exists to avoid. The scar: *never build the naive `O(|V|·d)` readout as a
default, even for a demo* — start from the tied/low-rank/VQ mitigation, because the moment the
readout matrix is the biggest thing in the model, you've rebuilt the vocabulary table you set out to
reclaim.
