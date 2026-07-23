# GROUND-UP-CONSTRAINTS.md — the charter

**Precedence: this charter outranks every phase document.** If a phase doc, a plan, or a clever
optimization conflicts with a constraint below, the constraint wins and the phase doc is wrong.
The charter exists to protect *experiment integrity* — the ability to attribute any number we
report to a specific, isolated cause.

---

## The failure this charter prevents

The default failure mode of "novel architecture" research is **borrowed competence**: a result
that looks like it came from the new idea but actually came from a pretrained embedding, a
subword tokenizer that already encodes the token paradigm, a GPU kernel that no one else can
reproduce, or a baseline chosen to lose. Borrowed competence is unfalsifiable — you cannot kill a
claim whose support is smuggled in from outside. Every constraint below closes one smuggling route.

---

## The constraints (C1–C8)

### C1 — No pretrained anything
No BERT, CLIP, SONAR, word2vec, GloVe, ImageBind, or any released checkpoint may appear in a
**controlled run**. The concept space is learned **from scratch**. Reason: a pretrained embedding
already contains the answer to "is there structure in language?", so using one to defend a novel
representation is circular. *Multimodal* pretrained encoders (CLIP/ImageBind/SONAR) may be
discussed as an *illustration* of the shared-concept-space idea but are **quarantined** (see below)
and never enter a headline number.

### C2 — Raw bytes/characters by default; subword is an ablation
The default input is **UTF-8 bytes** (256-symbol alphabet) or characters. A BPE/subword tokenizer
is a **named ablation**, never a default. Reason: BPE presupposes the discrete-token paradigm that
"pattern-as-token" is trying to replace; baking it in is a hidden variable that pre-loads the
conclusion. If a result only appears under BPE, that is a finding *about BPE*, reported as such.

### C3 — Minimal pure-Rust dependencies; no libtorch for headline numbers
Headline numbers run on **pure-Rust** tensor stacks (Burn with the NdArray/CPU backend for
correctness, Metal/WGPU for M1 speed; Candle as a lighter alternative). **No `tch`/libtorch,
no CUDA runtime** for any reported number. Reason: reproducibility and honesty about the compute
base — an M1 claim must be true on an M1, not on a borrowed A100. A libtorch/GPU comparison is
allowed only as an explicitly labeled, **quarantined perf note**.

### C4 — Determinism is correctness
Seed every RNG. Pin every dependency version. Log the **full config** (seed, versions, shapes,
hyperparameters, git SHA) alongside **every** number. Reason: a result that does not reproduce
**bit-for-bit on CPU** from its logged config is not a result — it is an anecdote. CPU
determinism is the gate; GPU nondeterminism (reduction order) is tolerated only for speed runs
that are separately re-verified on CPU.

### C5 — Strict train/val/test hygiene
Fixed splits, created before any tuning, with **no leakage** (no test text in train, no
val-tuned early-stopping evaluated on val and reported as test, no de-dup that peeks at test).
Reason: the cheapest way to fake a win is to let the test set influence training. Splits are
committed and hashed.

### C6 — Every mechanism switchable and independently ablated
Complex phase, unitary evolution, glow, sparse coding, and the pattern objective each have an
**on/off switch**. A reported effect must be produced by toggling exactly one switch. Reason:
only attributable numbers are worth reporting; a bundle of five ideas that wins together tells
you nothing about which idea did the work.

### C7 — The baseline is one you built
The comparison baseline is **built in this repo**, **param-matched**, and trained with the
**same loop, data, and budget**. Never an imported model, never a number copied from a paper's
table. Reason: baseline-selection is where most "novel architecture beats X" claims quietly die;
a fair baseline is the single most important honesty control (see doc 06).

### C8 — Inspectable over opaque
Prefer mechanisms whose internal state can be read out and audited (attractor patterns,
brightness counts, phase histograms, effective rank) over black-box performance. Reason: the
project's *product* thesis is a calibrated, inspectable confidence signal; if we cannot inspect
our own model, we cannot ship the thing we are claiming.

---

## Quarantine protocol (for any external reference)

Sometimes an external artifact is genuinely useful — as an *upper-bound sanity check*, a
*qualitative illustration*, or a *scaling reference*. It may be used **only** under quarantine:

1. **Label it** in the config as `quarantined: true` with the artifact name and version.
2. **Never** let it enter a controlled comparison or a headline number.
3. **Report the controlled (from-scratch) number first**, and the quarantined number only as a
   clearly separated "for reference" line.
4. **State the leak it would cause** if it were used un-quarantined (e.g. "CLIP already encodes
   visual-semantic alignment, so using it would answer C3 for free").
5. A quarantined artifact can *motivate* a design but can never *validate* one.

Example: "SONAR sentence embeddings are quarantined; we cite Large Concept Models as motivation
for pattern-as-token, but our controlled runs learn the pattern space from raw bytes."

---

## Reviewer checklist (run before any number is believed)

- [ ] **C1** Grep the run config for pretrained-checkpoint names → must be empty in controlled runs.
- [ ] **C2** Input is bytes/chars unless the run is explicitly labeled a BPE ablation.
- [ ] **C3** No `tch`/libtorch/CUDA in the dependency graph of the reported binary.
- [ ] **C4** The number ships with seed + versions + git SHA and reproduces bit-for-bit on CPU.
- [ ] **C5** Splits are fixed, hashed, and leak-checked; no val used as test.
- [ ] **C6** Exactly one switch differs between the two runs being compared.
- [ ] **C7** The baseline is built here, param-matched (±5%), same loop/data/budget.
- [ ] **C8** The winning mechanism's internal state was read out and looks sane, not just its metric.
- [ ] **Quarantine** Any external artifact is labeled, reported separately, and never in a headline.

---

### Interview questions this doc answers

- *"How do you stop yourself from cheating?"* Eight named constraints, each closing one smuggling
  route, plus a reviewer checklist that must pass before a number is believed.
- *"Why not just use a pretrained embedding to bootstrap?"* Because it makes the central claim
  unfalsifiable (C1): you can no longer tell whether the win came from the new idea or the import.
- *"Why bytes instead of BPE?"* BPE presupposes the very token paradigm we are testing against;
  it is a hidden variable, so it is an ablation, not a default (C2).

### Operator's scar

We once shipped a "3-point win" that survived for two weeks before someone noticed the baseline
used a smaller hidden size because of an off-by-one in the param-matching script. The win was
entirely the parameter gap. C7 and the "exactly one switch" rule in C6 are the direct scar tissue
from that: param-match first, verify the count, *then* run.
