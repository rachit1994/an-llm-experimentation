# 08 — Combinatorial capacity (the "permutations" idea, done right)

The intuition: *concepts are combinations of nodes, and combinations explode, so a small number of
nodes can represent astronomically many concepts.* This doc gives the verdict: **KEEP the
combinatorial insight, REJECT the naive counting** — because **naming a pattern ≠ storing it as a
reliable memory**, and those two are off by orders of magnitude.

---

## Part 1 — Naming capacity (the easy, true half)

**Target:** represent **36 billion** distinct patterns (a stand-in for "all the concepts a mind
might use"). How many nodes do you need under different codes?

| Coding scheme | Nodes needed for 36e9 patterns | Why |
|---|---|---|
| **Localist** (1 node per concept) | **36,000,000,000** | one dedicated node each — absurd |
| **Sparse, k=5 active (~1.5% of N)** | **≈ 339** | choose 5 of 339: `C(339,5) ≈ 3.6e10` |
| **Dense binary** (each node 0/1) | **36** | `2^36 ≈ 6.87e10 > 3.6e10` |
| **Analog, 16 levels/node** | **9** | `16^9 ≈ 6.87e10` |

The combinatorics are staggering and **real**: 36 nodes of dense binary, or ~339 nodes at 1.5%
sparsity, can *name* 36 billion patterns. Pushing further: a **2%-sparse, 512-node** code holds

```
C(512, ⌊0.02·512⌋) = C(512, 10) ≈ 3 × 10^20 patterns.
```

So the "permutations" intuition is **correct about naming**: a few hundred sparse nodes name more
patterns than there are concepts anyone will ever use. **Sparse distributed coding is the right
representation.**

---

## Part 2 — Storing capacity (the hard, crucial half)

Here is the trap. **Naming ≠ storing.** Naming asks "how many distinct patterns *exist*?"; storing
asks "how many can I write down as **reliable attractors** such that *bee always re-fires the same
pattern* under noise?" (that is the C3 / H4 "bee test"). Reliability costs weights, not just nodes.

**Classical Hopfield capacity** is the sobering number. A classical Hopfield network stores only

```
P_max ≈ 0.138 · N     (patterns, for N nodes, before catastrophic interference)
```

To reliably store **36 billion** patterns as classical Hopfield attractors:

```
N ≈ 36e9 / 0.138 ≈ 2.6 × 10^11  ≈  260 billion nodes.
```

**Infeasible.** So if "store concepts as attractors" meant *classical* Hopfield, the whole capacity
story would collapse — you'd need hundreds of billions of nodes to store what a few hundred can
*name*. This is exactly the gap between naming and storing.

---

## Part 3 — The resolution: modern Hopfield = attention (exponential capacity)

**Reference:** Ramsauer et al., *Hopfield Networks is All You Need*, arXiv:2008.02217.

The **modern** (continuous, "dense associative") Hopfield network has **exponential** storage
capacity in the dimension — not `0.138N`. Its update rule *is* the transformer attention operation:

```
retrieve(ξ) = X · softmax(β · Xᵀ ξ) ,    β = 1/√d
```

(`X` = stored patterns as columns, `ξ` = query, `β` = inverse-temperature/sharpness). One step of
this **is one attention step**. Because capacity is exponential in `d`, a **few thousand sparse
nodes** plus the associative weight matrix store **far more than 36 billion** reliable attractors —
and that weight matrix is **M1-sized** (doc 09: a `d×d` Hebbian matrix, ~1MB for `d=512`, roughly
`N²/2` weights, **independent of the number of stored patterns** because patterns *superpose* in the
same matrix).

So the corrected capacity story:

| Question | Naive answer | Correct answer |
|---|---|---|
| How many patterns can 512 sparse nodes **name**? | ~3e20 (C(512,10)) | ~3e20 ✓ |
| How many can classical Hopfield **store**? | "lots" | **only 0.138·N** (needs 260B nodes for 36B) ✗ |
| How many can **modern** Hopfield **store**? | — | **exponential in d** ⇒ a few thousand nodes suffice ✓ |

**KEEP:** sparse distributed coding + combinatorial naming + attractor storage.
**REJECT:** naive `C(N,k)` counting as if naming were storing, and classical `0.138N` as the storage
model. The right storage model is modern-Hopfield/attention.

---

## Part 4 — Concepts are never enumerated

A subtle but important consequence: **you never build a list of 36 billion concepts.** Concepts are
**implicit in combinations** of nodes; the model stores an **associative weight matrix** (superposed
patterns), not a codebook of every concept. "Bee," "wasp," and "bee-in-a-jar" are *points/attractors*
in the same `d`-dim space, retrieved on demand. This is why doc 09 says *never materialize 36B*: the
capacity lives in the superposition, not in an enumerated table. Enumerate nothing; superpose
everything.

---

## Part 5 — Multimodal "one screenshot → image/voice/text"

The vision — "read out one whole-state screenshot and render it as an image, a voice, or text" —
requires a **shared amodal concept space**: modality-specific **encoders** map pixels/audio/bytes
*into* the common pattern space, and modality-specific **decoders** map patterns *back out*. The
concepts themselves are amodal (the same "bee" attractor whether seen, heard, or read).

Pretrained multimodal systems (CLIP, ImageBind, SONAR) are the existence proof that a shared
concept space *works* — but they are **pretrained**, so under **C1 they are quarantined** (doc
`GROUND-UP-CONSTRAINTS.md`): they may **motivate** the design and appear as an upper-bound reference,
but in **controlled runs the encoders/decoders are trained from scratch** (Phase 6). The multimodal
screenshot is a *destination*, not a Phase-1 deliverable.

---

### Interview questions this doc answers

- *"Doesn't sparse coding give you unlimited capacity for free?"* For **naming**, nearly — 512 nodes
  at 2% sparsity name ~3e20 patterns. For **reliable storage**, no — classical Hopfield stores only
  0.138N (260B nodes for 36B patterns). Modern Hopfield (= attention) fixes this with exponential
  capacity.
- *"So do you build a table of all the concepts?"* Never. Concepts are implicit in node combinations
  and superposed in one associative matrix; you enumerate nothing.
- *"How would multimodality work?"* A shared amodal concept space with per-modality encoders/decoders
  trained from scratch; pretrained CLIP/ImageBind/SONAR are quarantined (C1).

### Operator's scar

The pitch deck once claimed "36 billion concepts in 36 nodes." It is true for *naming* and wildly
false for *storing* — and a Hopfield-literate reviewer caught it in ten seconds with "0.138N." We
almost shipped a headline that a first-year could falsify. The scar: *always say which verb you
mean — name or store — because the two differ by ~9 orders of magnitude, and the interesting,
defensible claim is the modern-Hopfield storage result, not the combinatorial naming trick.*
