# 01 — Prior art: quantum-*inspired* language models (with numbers)

This is not a green field. "Quantum-inspired" (classical math borrowing quantum *formalism*) is a
real, published subfield. The point of this doc is to give the honest numbers — the wins **and**
the walls — so the experiment is positioned against what already exists, not against a strawman.

---

## 1. Wave Network — token as complex vector (the closest prior art)

**Reference:** *Wave Network: An Ultra-Small Language Model*, arXiv:2411.02674.

**Idea.** Each token is a **complex-valued vector** where the **magnitude encodes global
semantics** (the token's position in the whole-sentence meaning) and the **phase encodes the
token↔context relationship**. Tokens are updated by **wave interference** and **wave modulation**
rather than dot-product attention.

**Numbers (AG News text classification):**

| Model | AG News accuracy | Notes |
|---|---|---|
| Single transformer layer + BERT embeddings | ~**71–72%** | single-layer reference |
| **Wave Network, single layer (interference)** | **90.91%** | ~**+19 pts** over the single transformer layer |
| **Wave Network, single layer (modulation)** | **91.66%** | ~**+20 pts** over the single transformer layer |
| BERT-base (full, pretrained) | **94.64%** | the upper reference it *approaches* |

Reported at **low VRAM and low training time** relative to the transformer reference. The headline
is *single-layer* competitiveness: a complex-vector layer with interference/modulation gets within
~3–4 points of full BERT-base on this task, and ~19–20 points above a single transformer layer.

**Backprop through it works.** *Token2Wave* (arXiv:2411.06989) works out the forward/backward
dynamics and gradient flow for Wave-Network-style complex token representations, so the approach is
trainable by standard autodiff, not a hand-tuned curiosity.

**Why it matters here.** This is the direct evidence for C1+C2 at *classification* scale: complex
amplitude + interference is not just expressible, it is *competitive* on a real task on modest
hardware. Our Phase-4 ablation (real-valued/phase=0) is designed to test whether that win survives
a fair, param-matched, from-scratch comparison (Wave Network used BERT embeddings for the
transformer reference; C1/C7 forbid that in our controlled runs).

---

## 2. Complex-valued quantum language models (C-NNQLM)

**Reference:** *A Complex-valued Neural Network Quantum Language Model*, ACM TOIS 2022,
doi:10.1145/3505138.

**Idea.** Words are **states in a Hilbert space**; a word's meaning is a **superposition of
sememes** (sub-meanings); composition uses **tensor products** and density-matrix mixing rather
than concatenation. The network is **complex-valued end to end**.

**Finding.** Across their language-modeling / matching benchmarks, the **complex-valued** model
**beats its real-valued** counterpart — direct evidence that the *complex* degree of freedom (i.e.
phase), not just "a bigger net," is doing work. This is the published analogue of our H1.

**Caveat we keep in view.** Their real-valued counterpart was not necessarily param-matched under
our C7 discipline; we treat "complex beats real" as *motivating*, and re-establish it ourselves in
Phase 4.

---

## 3. Tensor networks — MPS and exponential parameter reduction

**Reference:** Stoudenmire & Schwab, *Supervised Learning with Tensor Networks*, NeurIPS 2016,
arXiv:1605.05775.

**Idea.** Represent a high-order weight tensor as a **Matrix Product State (MPS)** — a chain of
small tensors contracted together. This is the same mathematical object (tensor networks) that
condensed-matter physics uses to compress quantum many-body states.

**Finding.** MPS gives **exponential parameter reduction** for models whose weight tensor has
low "bond dimension" — you replace an object exponential in the number of features with a chain
linear in it. It is the canonical demonstration that *quantum-formalism data structures* buy real
efficiency on classical hardware.

**Why it matters here.** It is the efficiency precedent behind doc 08's capacity argument and
doc 09's parameter budget: superposition/low-rank structure lets a small classical model *name*
astronomically many patterns.

---

## 4. Unitary RNNs — information-preserving recurrence

**Reference:** Arjovsky, Shah & Bengio, *Unitary Evolution Recurrent Neural Networks*, ICML 2016,
arXiv:1511.06464.

**Idea.** Constrain the recurrent matrix to be **unitary** (`U†U = I`), so the hidden state's norm
is preserved across time steps and gradients neither vanish nor explode.

**Finding.** The **uRNN solves the copy task and long-memory tasks that LSTMs fail** — it can carry
information across hundreds of steps because a unitary map loses no information by construction.

**Why it matters here.** This is the evidence for the unitary-evolution leg (C1, Phase 5): unitary
maps are not decoration, they are the known cure for long-range information loss. Our Phase-5 probe
(copy/recall) is a direct descendant of this task.

---

## Honest limits (the walls this field keeps hitting)

A fair prior-art section reports what has *not* worked, or works only under caveats:

1. **The generation / vocabulary wall.** Every published win above is on **classification or
   matching**, not open-ended **generation**. Born-rule readout over a full vocabulary is
   `O(|V|·d)` per step (doc 04), and no quantum-inspired model has convincingly beaten a
   transformer at large-vocabulary generation. This is our biggest honest gap (Phase 6 is gated).
2. **Transformers still win at scale.** These results are at **small scale** (single layers, small
   datasets). The evidence that complex/unitary/tensor structure keeps winning as you scale to
   billions of parameters and open generation is **absent**. We do not claim otherwise.
3. **Baseline-selection can manufacture wins.** The Wave Network transformer reference used BERT
   embeddings and a single layer; different baseline choices move the headline by many points.
   This is exactly the risk C7 (build your own param-matched baseline) exists to neutralize. A
   "quantum-inspired beats transformer" table is only as trustworthy as its baseline.
4. **Representational limits.** A pure unitary map is **linear and norm-preserving** — it cannot,
   by itself, do the nonlinear, contractive things (gating, forgetting, saturation) that real
   sequence models rely on. Practical models reintroduce nonlinearity, which partly gives back the
   simplicity the formalism promised. We treat "how much nonlinearity is needed" as an open
   empirical question, not a solved one.

---

## What the prior art licenses us to claim (and not claim)

- **Licensed:** complex amplitudes + interference are *competitive at classification scale on
  modest hardware*; complex beats real in at least one careful study; unitary recurrence cures
  long-range memory loss; tensor-network structure buys real parameter efficiency.
- **Not licensed:** any claim about *generation at scale*, any claim that these beat a *fairly
  matched* transformer once you build the baseline yourself, and any claim that requires quantum
  hardware.

---

### Interview questions this doc answers

- *"Has anyone done this before, and how did it go?"* Yes — Wave Network (90.91/91.66% single-layer
  AG News), C-NNQLM (complex beats real), MPS (exponential param reduction), uRNN (solves copy).
  All at small/classification scale; generation is the open wall.
- *"What's the strongest single number in the field?"* Wave Network single-layer **91.66%** on
  AG News (modulation), ~20 points over a single transformer layer, approaching BERT-base's 94.64%.
- *"Where do these results break?"* Generation over large vocab, scaling to billions of params, and
  baseline-selection — which is why we build our own baseline (C7).

### Operator's scar

We nearly cited the Wave Network 90.91% as *our* expected result in an early deck. It is not ours —
it used BERT embeddings for the reference and is a different (classification) task under different
fairness rules. The scar: *a number from someone else's table is a hypothesis about our result, not
a result.* Every external number in this repo is now tagged with its baseline and its task, and
none of them are allowed into our verdict tables (doc 05) until we reproduce the *comparison* under
C7.
