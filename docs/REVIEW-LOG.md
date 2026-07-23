# REVIEW-LOG.md — five adversarial review passes

Before any code, the dossier was put through five hostile reviews, each from a different seat. The
rule for each pass: *find the thing that makes this project embarrassing or wrong, and change the
docs so it can't happen.* Below is who reviewed, what they attacked, and **what changed** as a
result. Every change is traceable to a specific constraint or doc.

---

## Pass 1 — The QM physicist ("your quantum words are load-bearing lies")

**Attack.**
- "You keep saying *quantum*. Where is the entanglement? The measurement collapse? If it's all
  classical linear algebra, stop borrowing the mystique."
- "Barren plateaus, dequantization, and the simulability pincer mean a QC wouldn't even help you.
  Do you know that?"
- "Your 'whole-state screenshot' is *impossible* on real quantum hardware — measurement collapses
  the state."

**What changed.**
- Added **doc 02** in full (three walls + the collapse observation), and made "classical" the
  **first word** of the README verdict.
- Reframed the value of running classically as *positive*: the inspectable full-state screenshot is
  a property the **classical** implementation has and the quantum one **cannot** (doc 02,
  Observation 4; doc 10's inspectability).
- Banned "quantum" as a *standalone* selling point: the branding now lives or dies on the **H1
  phase=0 ablation** (docs 00, 05). If phase ties real at equal params, the word "quantum" is
  dropped. This pre-commitment is the physicist's demand made binding.

---

## Pass 2 — The ML researcher ("your baseline is doing the work")

**Attack.**
- "Wave Network's 90.91% used **BERT embeddings** for the transformer reference. You cannot cite
  that as your expected win — it's a different task with a favorable baseline."
- "Predict-your-own-latent objectives **collapse**. Where is your collapse metric? If your loss goes
  to zero, that's a *red flag*, not a result."
- "How are your comparisons param-matched and tuning-matched?"

**What changed.**
- **C7 (build-your-own param-matched baseline)** was hardened into an *automated* param counter
  (±5%) and equal-tuning-budget rule (doc 06 fairness rules).
- **Collapse instrumentation** (effective rank + per-dim variance) was made **line one of Phase 1**
  and elevated to a project-killing gate **G0** (docs 05, 07; H3). The "0.0002 loss = collapse"
  scar was written into doc 07.
- Every external number (Wave Network etc.) got a **task + baseline tag** and was **barred from
  verdict tables** until reproduced under C7 (doc 01 scar, references.md hygiene).

---

## Pass 3 — The systems / M1 engineer ("Metal will betray you")

**Attack.**
- "Apple Metal has **partial complex support**. Your complex kernels won't just run. Have you
  actually planned the backend story?"
- "Where do headline numbers come from — CPU or GPU? Because if it's GPU, reduction-order
  nondeterminism breaks your bit-for-bit reproducibility claim."
- "Have you *multiplied out* your memory budgets, or are they vibes?"

**What changed.**
- Adopted **`(re, im)` real pairs everywhere** so complex ops become structured real ops that run on
  both NdArray/CPU and Metal (docs 03, 04, 09; `implementation/00-...`). A unitary complex op = a
  structured orthogonal real op.
- Made **CPU-first the source of truth** (C3/C4): headline numbers are CPU, bit-for-bit
  reproducible; Metal is a separately-verified speed/Pareto note. GPU nondeterminism is quarantined.
- Rewrote **doc 09** so every memory line item shows its multiplication (the `1e6-codebook = 2 GB`
  and `1e6×512×4` scars), and defaulted to the **superposed `d×d` matrix** over enumerated tables.

---

## Pass 4 — The skeptic / VC ("what are you actually selling, and can you lose?")

**Attack.**
- "Are you going to beat GPT? Because if that's the pitch, you're dead. And if it isn't, say what
  the product *is*."
- "Every one of your claims sounds like it can only succeed. Show me the ones that can **fail** and
  what number kills them."
- "'36 billion concepts in 36 nodes' — is that even true?"

**What changed.**
- Wrote the **honest stakeholder framing** (doc 05): the sellable story is *small-scale efficiency +
  a novel representation + calibrated on-device confidence*, explicitly **NOT** a GPT head-to-head.
  Redefined "test against LLMs" as **mechanism parity → Pareto curve → labeled extrapolation**
  (doc 06), never a head-to-head, with the "cost axis is never optional" scar.
- Gave **every claim a kill number** and separated *branding-kill* (phase ties → drop "quantum")
  from *project-kill* (pattern collapses / >10% perplexity → stop) (docs 00, 05).
- Corrected the capacity claim: **KEEP combinatorial naming, REJECT naive storage** — classical
  Hopfield needs **260B nodes** for 36B patterns; modern Hopfield (= attention) fixes it (doc 08).
  The "name vs store" distinction became a headline rule.

---

## Pass 5 — The PM ("who is this for, and can they touch it?")

**Attack.**
- "The value has to reach a *user*. 'Novel representation' is not a benefit. What does a person
  *get*?"
- "You say 'inspectable.' Prove a non-researcher can read the confidence and act on it."
- "The most frequent token will dominate your 'glow' and the whole confidence story becomes a joke."

**What changed.**
- Reframed glow as the **on-device value story** (doc 10): a *calibrated, inspectable* confidence /
  abstain signal for a local model that **can't defer to a bigger cloud model** — the user-facing
  benefit is "it tells you when it's guessing."
- Made **confidence = retrieval margin (top1 − top2)** the concrete, printable number a
  non-researcher can threshold, and tied the whole claim to a **calibration kill number** (H5: beat
  no-glow on ECE, or narrow to salience) (docs 05, 10).
- Made the **three-part runaway fix MANDATORY** (log-compress + divisive/homeostatic normalization +
  IDF salience), with the "the model's most salient concept was a blank space" scar (doc 10) — so
  the confidence story can't invert into a punchline.

---

## Net effect of the five passes

| Pass | Seat | Biggest single change |
|---|---|---|
| 1 | QM physicist | doc 02 (why-not-QC) + "quantum" branding pre-committed to the H1 ablation |
| 2 | ML researcher | collapse instrumentation as line-one G0 gate; baseline fairness automated (C7) |
| 3 | systems/M1 | `(re,im)` pairs + CPU-first source-of-truth; memory budgets multiplied out |
| 4 | skeptic/VC | kill numbers for every claim; "no GPT head-to-head"; name-vs-store capacity fix |
| 5 | PM | glow reframed as inspectable on-device confidence; mandatory runaway fix |

---

### Interview questions this doc answers

- *"Did anyone stress-test this before you wrote code?"* Five adversarial passes (QM physicist, ML
  researcher, systems/M1, skeptic/VC, PM), each of which changed a specific doc or constraint.
- *"What did the reviews actually change?"* Added doc 02, made collapse a project-kill gate,
  adopted `(re,im)`/CPU-first, gave every claim a kill number and dropped any GPT head-to-head, and
  made the glow runaway fix mandatory.

### Operator's scar

The first review round was a single "looks good to me" from a friendly reviewer, and it caught
nothing — the collapse trap, the Metal caveat, and the frequency-runaway all survived it and would
have surfaced in production instead. The scar: *a review that doesn't change a document didn't
happen.* We now require each pass to name the seat it reviews from and to leave a diff; a pass with
no diff is re-run with a more hostile reviewer.
